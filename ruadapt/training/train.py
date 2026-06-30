"""Thin entrypoint: parse config → import factories → assemble → train.

Usage:
    python -m ruadapt.training.train --config config.json

    # Pre-tokenize and cache dataset (no GPU needed):
    python -m ruadapt.training.train --config config.json --data_prep_only

Or with torchrun for multi-GPU:
    torchrun --nproc_per_node=8 -m ruadapt.training.train --config config.json
"""

import json
import os
from typing import Any, Dict

import fire

from ruadapt.training.config.schema import (
    DataConfig,
    FreezeConfig,
    LoRAConfig,
    MainConfig,
    ModelConfig,
    SFTConfig,
    TrainingConfig,
    UnifiedDatasetConfig,
)
from ruadapt.training.core.distributed import cleanup_distributed, init_distributed, is_main_process
from ruadapt.training.core.freeze import (
    freeze_all_except,
    get_trainable_summary,
    register_embed_freeze_hook,
)
from ruadapt.training.core.model import apply_lora_with_tied_embeddings, load_model_and_tokenizer
from ruadapt.training.core.trainer import (
    EvaluateFirstStepCallback,
    SavePeftModelCallback,
    UnifiedTrainer,
    compute_wsd_steps,
)
from ruadapt.training.datasets.factory import load_factory
from ruadapt.training.datasets.debug import print_sample
from ruadapt.utils.seed import set_random_seed


def parse_config_from_json(config_path: str) -> MainConfig:
    """Parse MainConfig from a JSON file.

    The JSON structure maps to MainConfig fields:
    {
        "model": {"model_name_or_path": "...", ...},
        "data": {"train_file": "...", ...},
        "lora": {"peft": true, ...},
        "freeze": {"strategy": "none", ...},
        "training": {"output_dir": "...", ...},
        "dataset_factory": "my_module.MyFactory",
        "collator_factory": "my_module.MyCollatorFactory"
    }
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Validate top-level keys
    known_keys = {
        "model", "data", "lora", "freeze", "training",
        "unified_dataset", "sft", "dataset_factory", "collator_factory",
    }
    unknown = set(raw.keys()) - known_keys
    if unknown:
        raise ValueError(
            f"Unknown top-level keys in config: {unknown}. "
            f"Known keys: {known_keys}"
        )

    # Parse each sub-config
    model_config = ModelConfig(**raw.get("model", {}))
    data_config = DataConfig(**raw.get("data", {}))
    lora_config = LoRAConfig(**raw.get("lora", {}))
    freeze_config = FreezeConfig(**raw.get("freeze", {}))

    # TrainingConfig extends TrainingArguments — parse via its own mechanism
    training_dict = raw.get("training", {})
    training_config = _parse_training_config(training_dict)

    # Unified dataset config (optional)
    ud_raw = raw.get("unified_dataset")
    ud_config = UnifiedDatasetConfig(**ud_raw) if ud_raw else None

    # SFT config (optional)
    sft_raw = raw.get("sft")
    sft_config = SFTConfig(**sft_raw) if sft_raw else None

    return MainConfig(
        model=model_config,
        data=data_config,
        lora=lora_config,
        freeze=freeze_config,
        training=training_config,
        unified_dataset=ud_config,
        sft=sft_config,
        dataset_factory=raw.get("dataset_factory"),
        collator_factory=raw.get("collator_factory"),
    )


def _parse_training_config(raw: Dict[str, Any]) -> TrainingConfig:
    """Parse TrainingConfig from dict, handling both HF and custom fields."""
    from transformers import HfArgumentParser

    parser = HfArgumentParser((TrainingConfig,))
    (config,) = parser.parse_dict(raw)
    return config


def _data_prep_only(cfg: MainConfig):
    """Tokenize and cache dataset without loading full model."""
    from transformers import AutoTokenizer
    
    # Initialize random seed for reproducibility in data prep
    seed = getattr(cfg.training, "seed", 42)
    set_random_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name_or_path,
        trust_remote_code=cfg.model.trust_remote_code,
    )

    if not cfg.dataset_factory or not cfg.collator_factory:
        raise ValueError("dataset_factory and collator_factory required")

    ds_factory_cls = load_factory(cfg.dataset_factory)
    ds_factory = ds_factory_cls()

    if is_main_process():
        print("\n--- Train dataset ---")
    train_ds = ds_factory.create_train(tokenizer, cfg)

    if cfg.data.val_file:
        if is_main_process():
            print("\n--- Eval dataset ---")
        ds_factory.create_eval(tokenizer, cfg)

    if is_main_process():
        print("\n" + "=" * 60)
        print("  DATA PREPARATION COMPLETE")
        print("  Cache is ready. Run training without --data-prep-only.")
        print("=" * 60)


def main(config: str, data_prep_only: bool = False):
    """Main training entrypoint.

    Args:
        config: Path to JSON config file.
        data_prep_only: If True, only tokenize and cache dataset, then exit.
            No model loading, no training. Useful for pre-populating cache
            before multi-GPU runs.
    """
    # Parse config
    cfg = parse_config_from_json(config)

    # Init distributed
    init_distributed()

    # Initialize random seed globally for reproducibility across training libs (torch, numpy, random, etc)
    seed = getattr(cfg.training, "seed", 42)
    set_random_seed(seed)

    if data_prep_only:
        _data_prep_only(cfg)
        cleanup_distributed()
        return

    # Load model + tokenizer
    model, tokenizer, hf_config = load_model_and_tokenizer(cfg.model)

    # Enable gradient checkpointing (default for large models)
    if cfg.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # Apply freeze strategy
    if cfg.freeze.strategy == "embed_only":
        freeze_all_except(model, ["embed_tokens", "lm_head"])
        if cfg.freeze.freeze_idx is not None:
            register_embed_freeze_hook(model, cfg.freeze.freeze_idx)
    elif cfg.freeze.strategy == "custom" and cfg.freeze.unfreeze_modules:
        freeze_all_except(model, cfg.freeze.unfreeze_modules)

    # Apply LoRA
    if cfg.lora.peft:
        model = apply_lora_with_tied_embeddings(model, cfg.lora)

    # Print trainable summary
    if is_main_process():
        freeze_idx_val = cfg.freeze.freeze_idx if cfg.freeze.strategy == "embed_only" else None
        summary = get_trainable_summary(model, freeze_idx=freeze_idx_val)
        print(
            f"Trainable: {summary['trainable']:,} / {summary['total']:,} "
            f"({summary['percentage']:.2f}%)"
        )

    # Build dataset via factory
    if cfg.dataset_factory and cfg.collator_factory:
        ds_factory_cls = load_factory(cfg.dataset_factory)
        coll_factory_cls = load_factory(cfg.collator_factory)

        ds_factory = ds_factory_cls()
        coll_factory = coll_factory_cls()

        train_dataset = ds_factory.create_train(tokenizer, cfg)
        eval_dataset = ds_factory.create_eval(tokenizer, cfg)
        data_collator = coll_factory.create(tokenizer, cfg)
    else:
        raise ValueError(
            "dataset_factory and collator_factory must be specified in config. "
            "These are required for data preparation."
        )

    # Debug: print dataset samples
    if is_main_process():
        print_sample("TRAIN", train_dataset, tokenizer, num_samples=5)
        if eval_dataset is not None:
            print_sample("EVAL", eval_dataset, tokenizer, num_samples=5)

    # WSD scheduler: compute lr_scheduler_kwargs
    scheduler_type = getattr(cfg.training, "lr_scheduler_type", "")
    scheduler_val = scheduler_type.value if hasattr(scheduler_type, "value") else str(scheduler_type)
    if scheduler_val == "warmup_stable_decay":
        compute_wsd_steps(cfg, train_dataset)

    # Build callbacks
    callbacks = []
    if cfg.lora.peft:
        callbacks.append(SavePeftModelCallback())
    if getattr(cfg.training, "eval_on_start", False):
        callbacks.append(EvaluateFirstStepCallback())

    # Create trainer
    trainer = UnifiedTrainer(
        model=model,
        args=cfg.training,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if eval_dataset is not None else None,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    # Set label_names if not auto-detected
    if not trainer.label_names:
        trainer.label_names = ["labels"]

    # Resume from checkpoint if exists
    resume_from = None
    if cfg.training.output_dir and os.path.isdir(cfg.training.output_dir):
        from transformers.trainer_utils import get_last_checkpoint
        last_ckpt = get_last_checkpoint(cfg.training.output_dir)
        if last_ckpt is not None:
            resume_from = last_ckpt
            if is_main_process():
                print(f"Resuming from checkpoint: {last_ckpt}")

    # Train
    if is_main_process():
        print("Starting training...")

    trainer.train(resume_from_checkpoint=resume_from)

    # Save final model
    if is_main_process():
        print(f"Training finished. Saving to {cfg.training.output_dir}...")

    trainer.save_model(cfg.training.output_dir)

    if is_main_process():
        print("=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Final model saved to: {cfg.training.output_dir}")
        print("=" * 60)

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    fire.Fire(main)
