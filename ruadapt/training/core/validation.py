"""Validate training modes before loading weights or preparing datasets."""

from ruadapt.training.config.packing import resolve_packing
from ruadapt.training.core.packing_attention import SUPPORTED_BACKENDS, SUPPORTED_TYPES

PACKING_BACKENDS = SUPPORTED_BACKENDS
PACKING_MODEL_TYPES = SUPPORTED_TYPES


def validate_training_config(config):
    policy = resolve_packing(config)
    if policy.mode == "isolated":
        if config.model.attn_implementation not in PACKING_BACKENDS:
            raise ValueError("Isolated packing requires sdpa, flash_attention_2 or flash_attention_3")
        if config.training.remove_unused_columns:
            raise ValueError("Isolated packing requires remove_unused_columns=false")
    if config.freeze.strategy == "embed_only" and config.freeze.freeze_idx is not None:
        limit = config.freeze.freeze_idx
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError("freeze_idx must be a non-negative integer")
        if limit:
            if config.training.embed_weight_decay != 0:
                raise ValueError("Partial embedding freeze requires embed_weight_decay=0")
            # Flattened/sharded parameters do not preserve vocabulary row indexing.
            if config.training.fsdp or config.training.deepspeed:
                raise ValueError("Partial embedding freeze currently supports single-device/DDP, not FSDP/DeepSpeed")


def validate_packing_model(config, model):
    if resolve_packing(config).mode != "isolated":
        return
    model_config = model.config
    text_config = getattr(model_config, "text_config", model_config)
    if getattr(text_config, "model_type", None) not in PACKING_MODEL_TYPES:
        raise ValueError("Isolated packing supports Qwen3 and Qwen3.5 text layers")
    backend = getattr(text_config, "_attn_implementation", None)
    if backend not in PACKING_BACKENDS or backend != config.model.attn_implementation:
        raise ValueError(f"Loaded model uses {backend!r}, expected {config.model.attn_implementation!r}")
