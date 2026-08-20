"""UnifiedTrainer + callbacks: FSDP save, LoRA save, eval-first-step.

UnifiedTrainer extends HF Trainer with:
- FSDP checkpoint save via summon_full_params
- LoRA adapter save via SavePeftModelCallback
- Differential weight decay (embed_weight_decay vs global weight_decay)
- EvaluateFirstStepCallback (optional)
- WSD scheduler support (via lr_scheduler_kwargs)
"""

import os
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import Trainer, TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


class UnifiedTrainer(Trainer):
    """Trainer with FSDP save support and differential weight decay.

    Features:
    - Correct FSDP checkpoint saving via summon_full_params
    - Differential weight decay: embeddings get embed_weight_decay,
      other trainable params get global weight_decay
    - Optional separate eval collator (eval_collator kwarg): train and eval
      dataloaders may use different collation (e.g. packed train, dynamic-pad eval)
    - Works with FSDP, DDP, DeepSpeed, and single GPU
    """

    def __init__(self, *args, eval_collator=None, **kwargs):
        # transformers >= 5.x renamed 'tokenizer' to 'processing_class'
        if "tokenizer" in kwargs and "processing_class" not in kwargs:
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        super().__init__(*args, **kwargs)
        self.eval_collator = eval_collator

    def get_eval_dataloader(self, eval_dataset=None):
        """Build eval dataloader, optionally with a dedicated eval collator."""
        if self.eval_collator is None:
            return super().get_eval_dataloader(eval_dataset)
        orig_collator = self.data_collator
        self.data_collator = self.eval_collator
        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.data_collator = orig_collator

    def create_optimizer(self, model=None):
        """Create optimizer with differential weight decay groups.

        Groups:
        - decay_group: LoRA A/B weights, lm_head, biases/norms with dim < 2
            → global weight_decay
        - no_decay_group: embed_tokens (if trainable), biases, norms
            → embed_weight_decay (default 0.0)

        When LoRA is active and modules_to_save doesn't include embed_tokens,
        the embed group is empty (embed_tokens is frozen).
        """
        if self.optimizer is not None:
            return self.optimizer

        # Let HF Trainer create the base optimizer first to populate self.optimizer
        super().create_optimizer(model)

        # Now override with custom param groups if embed_weight_decay differs
        embed_wd = getattr(self.args, "embed_weight_decay", 0.0)
        if embed_wd == self.args.weight_decay:
            # No need to split — same WD for all params
            return self.optimizer

        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            return self.optimizer

        # Identify embed parameters (input + output embeddings)
        embed_params = set()
        input_embed = self.model.get_input_embeddings()
        if input_embed is not None:
            embed_params.update(id(p) for p in input_embed.parameters())
        output_embed = self.model.get_output_embeddings()
        if output_embed is not None:
            embed_params.update(id(p) for p in output_embed.parameters())

        # Split into groups
        decay_params = []
        no_decay_params = []
        for p in trainable_params:
            if id(p) in embed_params:
                no_decay_params.append(p)
            elif p.dim() >= 2:
                decay_params.append(p)
            else:
                no_decay_params.append(p)

        optimizer_groups = []
        if decay_params:
            optimizer_groups.append({
                "params": decay_params,
                "weight_decay": self.args.weight_decay,
            })
        if no_decay_params:
            optimizer_groups.append({
                "params": no_decay_params,
                "weight_decay": embed_wd,
            })

        if optimizer_groups:
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args, self.model
            )
            self.optimizer = optimizer_cls(optimizer_groups, **optimizer_kwargs)

        return self.optimizer

    # -- Checkpoint saving ---------------------------------------------------

    def _save_checkpoint(self, model, trial):
        """Save checkpoint with FSDP support."""
        if self._is_fsdp():
            self._save_fsdp_checkpoint(model, trial)
        else:
            super()._save_checkpoint(model, trial)

    def save_model(self, output_dir: str = None, _internal_call: bool = False):
        """Save model with FSDP support."""
        output_dir = output_dir or self.args.output_dir

        if self._is_fsdp():
            self._save_fsdp_model(output_dir)
        else:
            super().save_model(output_dir, _internal_call=_internal_call)

    def _is_fsdp(self) -> bool:
        """Check if model is wrapped with FSDP."""
        return isinstance(self.model, FSDP) or (
            hasattr(self.model, "module") and isinstance(self.model.module, FSDP)
        )

    def _save_fsdp_checkpoint(self, model, trial):
        """Save FSDP checkpoint: summon full params, save on rank 0."""
        checkpoint_folder = os.path.join(
            self.args.output_dir,
            f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}",
        )
        self._save_fsdp_model(checkpoint_folder)

        if self.args.process_index == 0:
            self.save_state()

    def _save_fsdp_model(self, output_dir: str):
        """Save FSDP model: summon full params, save on rank 0."""
        if self.args.process_index == 0:
            os.makedirs(output_dir, exist_ok=True)

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        with FSDP.summon_full_params(
            self.model, offload_to_cpu=True, rank0_only=True, writeback=False
        ):
            if self.args.process_index == 0:
                unwrapped_model.save_pretrained(output_dir)
                tok = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
                if tok is not None:
                    tok.save_pretrained(output_dir)


class SavePeftModelCallback(TrainerCallback):
    """Save LoRA adapter on checkpoint and training end."""

    def on_save(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self._save_adapter(args, state, kwargs)
        return control

    def on_train_end(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        peft_model_path = os.path.join(args.output_dir, "final_lora_adapter")
        kwargs["model"].save_pretrained(peft_model_path)
        tokenizer = kwargs.get("tokenizer") or kwargs.get("processing_class")
        if tokenizer is not None:
            tokenizer.save_pretrained(peft_model_path)
        return control

    def _save_adapter(self, args, state, kwargs):
        """Save adapter at checkpoint path."""
        if state.best_model_checkpoint is not None:
            checkpoint_folder = state.best_model_checkpoint
        else:
            checkpoint_folder = os.path.join(
                args.output_dir,
                f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}",
            )

        peft_model_path = os.path.join(checkpoint_folder, "lora_adapter")
        kwargs["model"].save_pretrained(peft_model_path)
        tokenizer = kwargs.get("tokenizer") or kwargs.get("processing_class")
        if tokenizer is not None:
            tokenizer.save_pretrained(peft_model_path)


class EvaluateFirstStepCallback(TrainerCallback):
    """Evaluate before the first training step (optional).

    Useful for getting baseline metrics before training starts.
    """

    def on_step_begin(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step == 0:
            control.should_evaluate = True
        return control


class LigerCheckCallback(TrainerCallback):
    """Print liger patch status at train start (rank 0).

    Liger is applied by HF Trainer inside train() (after __init__), so the
    check must run in on_train_begin. Expected values for Qwen3.5:
    base forward qualname starts with 'lce_forward', layer-0 MLP reports
    'LigerQwen3MoeSwiGLUMLP'.
    """

    def on_train_begin(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        model: Any = None,
        **kwargs,
    ):
        if args.process_index != 0 or model is None:
            return control
        unwrapped = model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        base = unwrapped.get_base_model() if hasattr(unwrapped, "get_base_model") else unwrapped
        fwd_name = getattr(getattr(base, "forward", None), "__qualname__", "?")
        layers = None
        for attr_path in ("model.layers", "model.language_model.layers"):
            obj = base
            try:
                for attr in attr_path.split("."):
                    obj = getattr(obj, attr)
                layers = obj
                break
            except AttributeError:
                continue
        mlp_name = layers[0].mlp._get_name() if layers else "?"
        print(f"[liger-check] base forward: {fwd_name}")
        print(f"[liger-check] layer0 mlp: {mlp_name}")
        return control


def compute_wsd_steps(cfg, train_dataset):
    """Compute WSD scheduler kwargs (num_stable_steps, num_decay_steps).

    Called when lr_scheduler_type == "warmup_stable_decay". Populates
    cfg.training.lr_scheduler_kwargs in-place.
    """
    from ruadapt.training.core.distributed import is_main_process

    n_gpus = dist.get_world_size() if dist.is_initialized() else 1
    t = cfg.training
    bs = t.per_device_train_batch_size
    accum = t.gradient_accumulation_steps
    effective_batch = bs * accum * n_gpus

    if t.max_steps and t.max_steps > 0:
        total_steps = t.max_steps
    else:
        total_steps = int(t.num_train_epochs * len(train_dataset) // effective_batch)

    # warmup_steps: in transformers 5.x, deprecated warmup_ratio gets assigned
    # to warmup_steps as a float. Handle both int and float cases.
    ws = t.warmup_steps
    if isinstance(ws, float) and 0 < ws < 1:
        warmup_steps = int(total_steps * ws)
    elif isinstance(ws, (int, float)) and ws > 0:
        warmup_steps = int(ws)
    elif hasattr(t, "warmup_ratio") and 0 < t.warmup_ratio < 1:
        warmup_steps = int(total_steps * t.warmup_ratio)
    else:
        warmup_steps = 0

    remaining = total_steps - warmup_steps
    wsd_frac = getattr(t, "wsd_constant_part", 0.85)
    t.lr_scheduler_kwargs = {
        "num_stable_steps": int(remaining * wsd_frac),
        "num_decay_steps": int(remaining * (1 - wsd_frac)) + 2,
    }

    if is_main_process():
        print(f"WSD scheduler: total={total_steps}, warmup={warmup_steps}, "
              f"stable={t.lr_scheduler_kwargs['num_stable_steps']}, "
              f"decay={t.lr_scheduler_kwargs['num_decay_steps']}")
