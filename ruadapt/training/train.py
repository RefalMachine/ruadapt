"""Thin entrypoint: parse config → import factories → assemble → train.

Usage:
    python -m ruadapt.training.train --config config.json

    # Pre-tokenize and cache dataset without loading model weights:
    python -m ruadapt.training.train --config config.json --data_prep_only

Or with torchrun for multi-GPU:
    torchrun --nproc_per_node=8 -m ruadapt.training.train --config config.json
"""

import json
import os
from typing import Any, Dict

from ruadapt.training.config.schema import (
    DataConfig,
    FreezeConfig,
    LoRAConfig,
    MainConfig,
    ModelConfig,
    SFTConfig,
    TrainingConfig,
    UnifiedDatasetConfig,
    PackingConfig,
    SequenceConfig,
)
from ruadapt.training.config.packing import resolve_packing, validate_legacy_packing_fields
from ruadapt.training.core.packing_attention import install_packing_attention
from ruadapt.training.core.distributed import cleanup_distributed, init_distributed, is_main_process, main_process_first
from ruadapt.training.core.freeze import (
    freeze_all_except,
    get_trainable_summary,
    register_embed_freeze_hook,
)
from ruadapt.training.core.model import apply_lora_with_tied_embeddings, load_model_and_tokenizer
from ruadapt.training.core.trainer import (
    EvaluateFirstStepCallback,
    LigerCheckCallback,
    UnifiedTrainer,
    compute_wsd_steps,
)
from ruadapt.training.datasets.collators import DocumentCollator
from ruadapt.training.datasets.factory import load_factory
from ruadapt.training.datasets.debug import print_sample
from ruadapt.training.core.validation import validate_training_config, validate_packing_model
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
        "unified_dataset", "sft", "dataset_factory", "collator_factory", "packing", "sequence",
    }
    unknown = set(raw.keys()) - known_keys
    if unknown:
        raise ValueError(
            f"Unknown top-level keys in config: {unknown}. "
            f"Known keys: {known_keys}"
        )

    validate_legacy_packing_fields(raw)
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
    sft_config = SFTConfig(**sft_raw) if sft_raw is not None else None

    return MainConfig(
        model=model_config,
        data=data_config,
        lora=lora_config,
        freeze=freeze_config,
        training=training_config,
        packing=PackingConfig(**raw["packing"]) if raw.get("packing") is not None else None,
        sequence=SequenceConfig(**raw["sequence"]) if raw.get("sequence") is not None else None,
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


def _data_prep_only(cfg: MainConfig, prepared_dataset_dir=None):
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
    train_dataset = ds_factory.create_train(tokenizer, cfg)

    eval_dataset = None
    if cfg.data.val_file:
        if is_main_process():
            print("\n--- Eval dataset ---")
        eval_dataset = ds_factory.create_eval(tokenizer, cfg)

    if prepared_dataset_dir:
        from ruadapt.training.datasets.prepared import save_prepared
        save_prepared(prepared_dataset_dir, cfg, train_dataset, eval_dataset)
        print(f"Prepared dataset manifest: {prepared_dataset_dir}", flush=True)

    if is_main_process():
        print("\n" + "=" * 60)
        print("  DATA PREPARATION COMPLETE")
        print("  Cache is ready. Run training without --data-prep-only.")
        print("=" * 60)


def main(config: str, data_prep_only: bool = False, prepared_dataset_dir: str = None):
    """Main training entrypoint.

    Args:
        config: Path to JSON config file.
        data_prep_only: If True, only tokenize and cache dataset, then exit.
            No model loading, no training. Useful for pre-populating cache
            before multi-GPU runs.
        prepared_dataset_dir: With data_prep_only, export references to ready Arrow
            files here. Set data.prepared_dataset_dir to load them during training.
    """
    if prepared_dataset_dir and not data_prep_only:
        raise ValueError("--prepared_dataset_dir is an export option for --data_prep_only")
    # Parse config
    cfg = parse_config_from_json(config)
    validate_training_config(cfg)
    packing_policy = resolve_packing(cfg)

    # Init distributed
    init_distributed()

    # Initialize random seed globally for reproducibility across training libs (torch, numpy, random, etc)
    seed = getattr(cfg.training, "seed", 42)
    set_random_seed(seed)

    if data_prep_only:
        _data_prep_only(cfg, prepared_dataset_dir)
        cleanup_distributed()
        return

    prepared_datasets = None
    if cfg.data.prepared_dataset_dir:
        from ruadapt.training.datasets.prepared import load_prepared
        prepared_datasets = main_process_first(lambda: load_prepared(cfg.data.prepared_dataset_dir, cfg))
        if is_main_process():
            train_ready, eval_ready = prepared_datasets
            print(f"[data] Prepared Arrow loaded: train={len(train_ready):,}, "
                  f"eval={len(eval_ready) if eval_ready is not None else 0:,}; preprocessing skipped", flush=True)

    # Load model + tokenizer
    model, tokenizer, hf_config = load_model_and_tokenizer(cfg.model)
    validate_packing_model(cfg, model)
    if packing_policy.mode == "isolated":
        install_packing_attention(model)

    # Enable gradient checkpointing (default for large models)
    if cfg.training.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()

    # Apply freeze strategy
    if cfg.freeze.strategy == "embed_only":
        freeze_all_except(model, ["embed_tokens", "lm_head"])
    elif cfg.freeze.strategy == "custom" and cfg.freeze.unfreeze_modules:
        freeze_all_except(model, cfg.freeze.unfreeze_modules)

    # Apply LoRA
    if cfg.lora.peft:
        model = apply_lora_with_tied_embeddings(model, cfg.lora)

    if cfg.freeze.strategy == "embed_only" and cfg.freeze.freeze_idx is not None:
        register_embed_freeze_hook(model, cfg.freeze.freeze_idx)

    # FP8 storage fail-fast after PEFT: every quantized base layer must still be
    # patched (adapters wrap them but call base_layer.forward), no fp8 param trainable.
    if cfg.model.fp8_storage:
        from ruadapt.training.core.fp8 import verify_fp8_storage

        if not cfg.lora.peft:
            raise ValueError("model.fp8_storage=true requires lora.peft=true (frozen fp8 base)")
        summary = verify_fp8_storage(model)
        print(
            f"[fp8-storage] params GiB after PEFT: fp8_frozen={summary['fp8_frozen']:.1f}, "
            f"bf16_frozen={summary['bf16_frozen']:.1f}, trainable={summary['trainable']:.1f}"
        )
        # HF Trainer rejects fp8 quantization for training (FineGrainedFP8HfQuantizer
        # is inference-only: is_trainable=False). fp8_storage keeps the base frozen and
        # computes through patched bf16 forwards, so mark the PEFT config as loaded to
        # skip validate_quantization_for_training's "quantization does not support
        # training" branch (same flag transformers itself checks).
        model._hf_peft_config_loaded = True
        if cfg.training.gradient_checkpointing:
            print(
                "[fp8-storage] WARNING: gradient_checkpointing=true with fp8_storage — "
                "dequantization runs twice per step. The point of fp8_storage is to fit "
                "without checkpointing (see ANALYSYS.md, section 8)."
            )

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

        # Rank-gated dataset preparation: rank 0 builds first and populates the
        # HF-datasets cache, other ranks wait at the barrier then read from cache.
        # Avoids N ranks recomputing the same tokenization/packing maps in parallel.
        def _build_datasets():
            if prepared_datasets is not None:
                return prepared_datasets
            return ds_factory.create_train(tokenizer, cfg), ds_factory.create_eval(tokenizer, cfg)

        train_dataset, eval_dataset = main_process_first(_build_datasets)
        data_collator = coll_factory.create(tokenizer, cfg)

        # Packing mode: eval stays unpacked and keeps v7 dynamic-pad semantics
        eval_collator = None
        if hasattr(coll_factory, "create_eval"):
            eval_collator = coll_factory.create_eval(tokenizer, cfg)
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
    if getattr(cfg.training, "eval_on_start", False):
        callbacks.append(EvaluateFirstStepCallback())
    if cfg.training.use_liger_kernel:
        callbacks.append(LigerCheckCallback())

    # Create trainer
    trainer = UnifiedTrainer(
        model=model,
        args=cfg.training,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if eval_dataset is not None else None,
        data_collator=data_collator,
        eval_collator=eval_collator,
        processing_class=tokenizer,
        packing_policy=packing_policy,
        callbacks=callbacks,
    )

    # Set label_names if not auto-detected
    if not trainer.label_names:
        trainer.label_names = ["labels"]

    # Packing fail-fast guards (see ANALYSYS.md, section 6.5):
    # HF Trainer silently drops dataset columns absent from model.forward
    # signature when remove_unused_columns=True — seq_idx/cu_seq_lens_q would
    # vanish and GatedDeltaNet layers would silently lose document isolation.
    if packing_policy.mode == "isolated":
        if trainer.args.remove_unused_columns:
            raise ValueError(
                "Isolated packing requires training.remove_unused_columns=false, "
                "otherwise HF Trainer silently drops seq_idx/cu_seq_lens_q/position_ids "
                "columns and document isolation breaks."
            )
        sample_batch = next(iter(trainer.get_train_dataloader()))
        missing = [k for k in DocumentCollator.REQUIRED_KEYS if k not in sample_batch]
        if missing:
            raise ValueError(
                f"Packed dataloader batch is missing keys {missing}. "
                "Document isolation would break. Check dataset factory output "
                "and remove_unused_columns."
            )
        if is_main_process():
            ids, labels = sample_batch["input_ids"], sample_batch["labels"]
            print(f"[packing] logical_B={cfg.training.per_device_train_batch_size}, "
                  f"physical_shape={tuple(ids.shape)}, T={ids.numel()}, "
                  f"targets={labels[..., 1:].ne(-100).sum().item()}, "
                  f"max_segment={sample_batch['max_length_q']}")

    if is_main_process():
        print(f"[packing] resolved policy: {packing_policy.to_dict()}; backend={cfg.model.attn_implementation}")

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
    '''
    from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN
    print(model.config.model_type)                        # скорее всего "qwen3_5"
    print(model.config.model_type in MODEL_TYPE_TO_APPLY_LIGER_FN)

    # 2. Пропатчен ли forward (после создания trainer)
    print(model.forward.__qualname__, model.forward.__module__)

    import torch

    B, L = 2, 2048
    for _ in range(2):
        fake = {
            "input_ids": torch.randint(0, 248000, (B, L), device="cuda"),
            "attention_mask": torch.ones(B, L, dtype=torch.long, device="cuda"),
            "labels": torch.where(
                torch.rand(B, L, device="cuda") > 0.5,
                torch.randint(0, 248000, (B, L), device="cuda"),
                torch.tensor(-100, device="cuda"),
            ),
        }
        out = trainer.model(**fake)
        out.loss.backward()
        trainer.model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    '''
    
    print(os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
    trainer.train(resume_from_checkpoint=resume_from)

    # Save final model
    if is_main_process():
        print(f"Training finished. Saving to {cfg.training.output_dir}...")

    trainer.save_model(cfg.training.output_dir)
    if cfg.lora.peft:
        trainer.export_final_adapter(cfg.training.output_dir)

    if is_main_process():
        print("=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Final model saved to: {cfg.training.output_dir}")
        print("=" * 60)

    # Cleanup
    cleanup_distributed()


def cli():
    """Shared console and python -m entry point."""
    import fire

    fire.Fire(main)


if __name__ == "__main__":
    cli()
