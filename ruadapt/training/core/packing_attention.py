"""Instance-local packing integration for Qwen3 and Qwen3.5 text training.

    SDPA: build the block causal mask once per forward; inject it ONLY into
    full-attention modules. The model/FLA padding path never receives a 4D mask.
    FA2/FA3: native HF varlen kwargs. All backends share FLA boundary metadata.

Hooks add no parameters or state-dict keys and do not alter unpacked forwards.
"""

import torch

SUPPORTED_TYPES = {"qwen3", "qwen3_5", "qwen3_5_text"}
SUPPORTED_BACKENDS = {"sdpa", "flash_attention_2", "flash_attention_3"}


def block_causal_mask(seq_ids, *, window=None):
    """Boolean SDPA mask [1, 1, T, T]; True means allowed, including the diagonal."""
    ids = seq_ids.reshape(-1)
    t = torch.arange(ids.numel(), device=ids.device)
    mask = ids[:, None] == ids[None, :]
    mask &= t[:, None] >= t[None, :]
    if window is not None:
        mask &= t[:, None] < t[None, :] + window
    return mask[None, None]


def _prepare_forward(module, args, kwargs):
    if "_packing_lengths" in kwargs:
        kwargs = dict(kwargs)
        kwargs.pop("_packing_lengths")
    if kwargs.get("cu_seq_lens_q") is None:
        return args, kwargs
    kwargs = dict(kwargs)
    if kwargs.get("past_key_values") is not None or kwargs.get("use_cache") is True:
        raise ValueError("Isolated training does not support state/cache carried between batches")
    kwargs["use_cache"] = False
    if kwargs.get("attention_mask") is not None:
        raise ValueError("Isolated packing uses segment metadata, not a model-level padding mask")
    if kwargs["input_ids"].shape[0] != 1:
        raise ValueError("Isolated packs must be flattened by DocumentCollator")
    for linear in module._ruadapt_linear_layers:
        if linear.causal_conv1d_fn is None or linear.chunk_gated_delta_rule.__module__.startswith("transformers"):
            raise ValueError("Qwen3.5 isolated packing requires FLA and causal-conv1d kernels")
        if kwargs["input_ids"].device.type != "cuda":
            raise ValueError("Qwen3.5 isolated packing requires CUDA for FLA/convolution")
    if module._ruadapt_packing_backend == "sdpa":
        kwargs["_ruadapt_block_mask"] = block_causal_mask(kwargs["seq_idx"])
    return args, kwargs


def _prepare_attention(module, args, kwargs):
    mask = kwargs.get("_ruadapt_block_mask")
    if mask is None:
        return args, kwargs
    kwargs = dict(kwargs)
    kwargs.pop("_ruadapt_block_mask")
    window = getattr(module, "sliding_window", None)
    if window is not None:
        mask = block_causal_mask(kwargs["seq_idx"], window=window)
    kwargs["attention_mask"] = mask
    kwargs["is_causal"] = False  # causal condition is already in the block mask
    return args, kwargs


def install_packing_attention(model):
    """Install only on a supported model instance, before PEFT/Trainer wrapping."""
    if hasattr(model, "_ruadapt_packing_backend"):
        return
    config = getattr(model.config, "text_config", model.config)
    if config.model_type not in SUPPORTED_TYPES or config._attn_implementation not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported isolated packing: {config.model_type}/{config._attn_implementation}")
    attention = [m for m in model.modules() if type(m).__name__ in {"Qwen3Attention", "Qwen3_5Attention"}]
    if not attention:
        raise ValueError("No supported text attention modules found; check Transformers version")
    # A tuple does not register another module tree/state-dict path.
    model._ruadapt_linear_layers = tuple(m for m in model.modules() if type(m).__name__ == "Qwen3_5GatedDeltaNet")
    model._ruadapt_packing_backend = config._attn_implementation
    model.register_forward_pre_hook(_prepare_forward, with_kwargs=True)
    for module in attention:
        module.register_forward_pre_hook(_prepare_attention, with_kwargs=True)
