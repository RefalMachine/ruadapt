"""UnifiedTrainer: native HF checkpoints, partial freeze and eval collation.

UnifiedTrainer extends HF Trainer with:
- Native HF checkpoint/resume for adapters and distributed models
- Differential weight decay (embed_weight_decay vs global weight_decay)
- EvaluateFirstStepCallback (optional)
- WSD scheduler support (via lr_scheduler_kwargs)
"""

import json
import os
import shutil
import warnings
from typing import Any

import torch
import torch.distributed as dist
from transformers import Trainer, TrainerCallback, TrainerControl, TrainerState

from ruadapt.training.core.freeze import frozen_parameter_limits, protect_frozen_optimizer_rows


class UnifiedTrainer(Trainer):
    """Trainer with FSDP save support and differential weight decay.

    Features:
    - Native HF/Accelerate checkpoint saving, including optimizer and RNG state
    - Differential weight decay: embeddings get embed_weight_decay,
      other trainable params get global weight_decay
    - Optional separate eval collator (eval_collator kwarg): train and eval
      dataloaders may use different collation (e.g. packed train, dynamic-pad eval)
    - Works with FSDP, DDP, DeepSpeed, and single GPU
    """

    def __init__(self, *args, eval_collator=None, packing_policy=None, **kwargs):
        # transformers >= 5.x renamed 'tokenizer' to 'processing_class'
        if "tokenizer" in kwargs and "processing_class" not in kwargs:
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        super().__init__(*args, **kwargs)
        self.eval_collator = eval_collator
        self.packing_policy = packing_policy
        self.token_normalized_loss = packing_policy is not None and packing_policy.loss_normalization == "tokens"
        if self.token_normalized_loss:
            if self.label_smoother is not None or self.compute_loss_func is not None:
                raise ValueError("Token normalization requires the model's causal-LM loss (no label smoothing/custom loss)")
            self.model_accepts_loss_kwargs = True
            self.args.average_tokens_across_devices = True

    def _get_num_items_in_batch(self, batch_samples, device):
        if not self.token_normalized_loss:
            return super()._get_num_items_in_batch(batch_samples, device)
        # HF counts unshifted labels; causal LM never predicts column zero.
        shifted = [{**batch, "labels": batch["labels"][..., 1:]} for batch in batch_samples]
        count = super()._get_num_items_in_batch(shifted, device)
        return count.clamp_min(1) if torch.is_tensor(count) else max(count or 0, 1)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if "_packing_lengths" in inputs:
            inputs = {k: v for k, v in inputs.items() if k != "_packing_lengths"}
        if self.token_normalized_loss and num_items_in_batch is None:
            # Evaluation/direct calls: no accumulation-window denominator. Do
            # not ask super() to apply the training world-size compensation.
            inputs = {**inputs, "num_items_in_batch": inputs["labels"][..., 1:].ne(-100).sum().clamp_min(1)}
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        pack_lengths = inputs.get("_packing_lengths")
        if pack_lengths is not None:
            inputs = {k: v for k, v in inputs.items() if k != "_packing_lengths"}
        if not self.token_normalized_loss or not hasattr(self, "_eval_token_totals"):
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        loss, logits, labels = super().prediction_step(model, inputs, False, ignore_keys)
        original_logits, original_labels = logits, labels
        if labels is not None and logits is not None:
            logits = logits[0] if isinstance(logits, tuple) else logits
            # One item per DATASET sample for gather_for_metrics: it removes
            # duplicated last-batch samples from Accelerate's distributed sampler.
            if "cu_seq_lens_q" in inputs:
                lengths = (pack_lengths.tolist() if pack_lengths is not None else
                           [self.packing_policy.max_length] * (labels.numel() // self.packing_policy.max_length))
                logits = logits.squeeze(0).split(lengths)
                labels = labels.squeeze(0).split(lengths)
            stats = []
            for pred, target in zip(logits, labels):
                target = target[1:]
                total = torch.zeros((), dtype=torch.float64, device=pred.device)
                # Bound the extra fp32 softmax workspace for large vocabularies.
                for start in range(0, target.numel(), 128):
                    total += torch.nn.functional.cross_entropy(
                        pred[start:start + 128][:target[start:start + 128].numel()].float(),
                        target[start:start + 128], reduction="sum", ignore_index=-100,
                    ).double()
                stats.append(torch.stack((total, target.ne(-100).sum().double())))
            gathered = self.accelerator.gather_for_metrics(torch.stack(stats))
            self._eval_token_totals += gathered.sum(0).cpu()
        return (loss, None, None) if prediction_loss_only else (loss, original_logits, original_labels)

    def evaluation_loop(self, *args, **kwargs):
        if not self.token_normalized_loss:
            return super().evaluation_loop(*args, **kwargs)
        self._eval_token_totals = torch.zeros(2, dtype=torch.float64)
        try:
            output = super().evaluation_loop(*args, **kwargs)
            total, count = self._eval_token_totals.tolist()
            prefix = kwargs.get("metric_key_prefix", "eval")
            output.metrics[f"{prefix}_loss"] = total / max(count, 1)
            output.metrics[f"{prefix}_target_count"] = int(count)
            return output
        finally:
            del self._eval_token_totals

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
            protect_frozen_optimizer_rows(self.model, self.optimizer)
            return self.optimizer

        # Let HF Trainer create the base optimizer first to populate self.optimizer
        super().create_optimizer(model)

        # Now override with custom param groups if embed_weight_decay differs
        embed_wd = getattr(self.args, "embed_weight_decay", 0.0)
        if embed_wd == self.args.weight_decay:
            # No need to split — same WD for all params
            protect_frozen_optimizer_rows(self.model, self.optimizer)
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

        protect_frozen_optimizer_rows(self.model, self.optimizer)
        return self.optimizer

    # -- Checkpoint saving ---------------------------------------------------

    def _save(self, output_dir=None, state_dict=None):
        # Trainer owns rank gating, state gathering and optimizer/RNG checkpoints.
        super()._save(output_dir, state_dict=state_dict)
        with open(os.path.join(output_dir or self.args.output_dir, "ruadapt_freeze.json"), "w") as f:
            json.dump(frozen_parameter_limits(self.model), f, sort_keys=True)
        if self.packing_policy is not None:
            with open(os.path.join(output_dir or self.args.output_dir, "ruadapt_packing.json"), "w") as f:
                json.dump(self.packing_policy.to_dict(), f, sort_keys=True)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        model = model if model is not None else self.model
        packing_file = os.path.join(resume_from_checkpoint, "ruadapt_packing.json")
        if self.packing_policy is not None and os.path.isfile(packing_file):
            with open(packing_file) as f:
                if not self.packing_policy.matches_checkpoint(json.load(f)):
                    raise ValueError("Checkpoint packing/loss policy differs; resume aborted")
        elif self.packing_policy is not None:
            warnings.warn("Legacy checkpoint has no packing/loss metadata; verify data and normalization before resume")
        freeze_file = os.path.join(resume_from_checkpoint, "ruadapt_freeze.json")
        if os.path.isfile(freeze_file):
            with open(freeze_file) as f:
                saved_limits = json.load(f)
            if saved_limits != frozen_parameter_limits(model):
                raise ValueError("Checkpoint freeze configuration differs from the current model; resume aborted")
        elif frozen_parameter_limits(model):
            warnings.warn("Legacy checkpoint has no freeze metadata; previously changed rows cannot be recovered")

        # Old RuAdapt checkpoints contain a duplicate lora_adapter directory.
        # HF interprets it as a separately named adapter and fails to load the
        # default adapter from the root. Prefer the canonical root for this case.
        from peft import PeftModel

        root_adapter = os.path.isfile(os.path.join(resume_from_checkpoint, "adapter_config.json")) and any(
            os.path.isfile(os.path.join(resume_from_checkpoint, name))
            for name in ("adapter_model.safetensors", "adapter_model.bin")
        )
        if (
            isinstance(model, PeftModel)
            and not self.is_fsdp_enabled
            and not self.is_deepspeed_enabled
            and root_adapter
            and os.path.isdir(os.path.join(resume_from_checkpoint, "lora_adapter"))
            and set(model.peft_config) == {"default"}
        ):
            model.load_adapter(resume_from_checkpoint, "default", is_trainable=True)
            model.set_adapter("default")
            return
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)

    def export_final_adapter(self, output_dir=None):
        """Copy the gathered final adapter for legacy consumers, on the writer only.

        Call after save_model on all ranks. Never create a duplicate adapter
        subdirectory in checkpoint-N: HF treats such directories as adapters.
        """
        if not self.args.should_save:
            return
        source = output_dir or self.args.output_dir
        if not os.path.isfile(os.path.join(source, "adapter_config.json")):
            return  # e.g. a sharded-only checkpoint, not an exported adapter
        if not any(os.path.isfile(os.path.join(source, n)) for n in ("adapter_model.safetensors", "adapter_model.bin")):
            return
        destination = os.path.join(source, "final_lora_adapter")
        os.makedirs(destination, exist_ok=True)
        # Save tokenizer via its API to include custom vocabulary file formats.
        if self.processing_class is not None:
            self.processing_class.save_pretrained(destination)
        for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin", "README.md"):
            path = os.path.join(source, name)
            if os.path.isfile(path):
                shutil.copy2(path, os.path.join(destination, name))


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
