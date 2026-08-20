"""Model + tokenizer loading, architecture detection, LoRA with tied embeddings.

Handles:
- Auto-detection of model architecture (Qwen3.5, etc.)
- bf16/fp16 dtype
- flash_attention_2 / sdpa / eager
- resize_token_embeddings
- LoRA application with tie_word_embeddings workaround
"""

from typing import Optional, Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from ruadapt.training.config.schema import LoRAConfig, ModelConfig
from ruadapt.training.core.fp8 import apply_fp8_storage, set_dequant_compiled, verify_fp8_storage

_TEXT_PREFIX = "model.language_model."


def _rewrite_modules_to_not_convert(names, text_only: bool):
    """Rewrite checkpoint-level module names for the text-only (CausalLM) class.

    The FP8 checkpoint lists exclusions with the multimodal prefix
    (model.language_model.X / model.visual.X / mtp.X). Qwen3_5ForCausalLM's
    module tree has no language_model/visual/mtp segments, and
    should_convert_module matches patterns anchored at the name start, so the
    original list would silently fail to exclude conv1d/in_proj_a/in_proj_b —
    they would be replaced by FP8Linear and weight loading would break.
    """
    out = []
    for name in names or []:
        if name.startswith(_TEXT_PREFIX):
            out.append("model." + name[len(_TEXT_PREFIX):])
        elif name.startswith(("model.visual.", "mtp.")):
            if not text_only:
                out.append(name)
        else:
            out.append(name)
    return out


def _qc_get(qc, key, default=None):
    """Read a quantization_config field from either a dict or an object."""
    if qc is None:
        return default
    if isinstance(qc, dict):
        return qc.get(key, default)
    return getattr(qc, key, default)


def _build_fp8_quantization_config(hf_config, text_only: bool):
    """Build an explicit FineGrainedFP8Config from the checkpoint's quantization_config."""
    from transformers import FineGrainedFP8Config

    raw = getattr(hf_config, "quantization_config", None)
    if raw is None or _qc_get(raw, "quant_method") != "fp8":
        raise ValueError(
            "model.fp8_storage=true requires a checkpoint with "
            "quantization_config.quant_method == 'fp8' (HF finegrained fp8)"
        )
    block_size = _qc_get(raw, "weight_block_size") or [128, 128]
    return FineGrainedFP8Config(
        activation_scheme=_qc_get(raw, "activation_scheme", "dynamic"),
        weight_block_size=tuple(block_size),
        modules_to_not_convert=_rewrite_modules_to_not_convert(
            _qc_get(raw, "modules_to_not_convert"), text_only
        ),
    )


def load_model_and_tokenizer(
    config: ModelConfig,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, AutoConfig]:
    """Load model, tokenizer, and config from ModelConfig.

    Handles architecture detection (Qwen3.5 conditional), dtype, and attention impl.

    Returns:
        (model, tokenizer, hf_config)
    """
    hf_config = AutoConfig.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
    )

    # Determine torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": "auto",
    }
    torch_dtype = dtype_map.get(config.torch_dtype, torch.bfloat16)

    # Detect architecture — Qwen3.5 conditional needs explicit class
    model_cls = AutoModelForCausalLM
    architectures = getattr(hf_config, "architectures", []) or []
    if "Qwen3_5ForConditionalGeneration" in architectures:
        if config.text_only:
            # Text-only training: skip the vision tower entirely.
            # Qwen3_5ForCausalLM ignores ^mtp.* and ^model.visual.* keys on load.
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

            model_cls = Qwen3_5ForCausalLM
        else:
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5ForConditionalGeneration,
            )

            model_cls = Qwen3_5ForConditionalGeneration

    # Load model
    from_pretrained_kwargs = {}
    if config.fp8_storage:
        # Explicit config (instead of auto-detect) so text_only gets exclusion
        # names matching the CausalLM module tree, see _rewrite_modules_to_not_convert.
        from_pretrained_kwargs["quantization_config"] = _build_fp8_quantization_config(
            hf_config, config.text_only
        )
    model = model_cls.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=config.attn_implementation,
        trust_remote_code=config.trust_remote_code,
        **from_pretrained_kwargs,
    )

    if config.fp8_storage:
        # QLoRA-style: keep frozen weights in fp8, compute in bf16 via patched
        # dequant forwards (autograd-friendly; HF's native FP8Linear forward has
        # no backward). Must run before PEFT wrapping so adapters see patched layers.
        apply_fp8_storage(model)
        if config.fp8_compile_dequant:
            set_dequant_compiled(True)
        # The GiB summary is printed in train.py after PEFT: pre-PEFT
        # "trainable" contains embed_tokens/lm_head, not adapters, and the log
        # misreads (FP8_CHECK.md, D8).
        verify_fp8_storage(model)

    if model_cls.__name__ == "Qwen3_5ForCausalLM":
        # Qwen3_5ForConditionalGeneration declares accepts_loss_kwargs=False, but
        # Qwen3_5ForCausalLM has no such attribute. Without it, HF Trainer falls
        # back to forward-signature inspection (**kwargs present -> True) and starts
        # passing num_items_in_batch, silently switching loss normalization from
        # per-microbatch mean to global token mean (see ANALYSYS.md, section 4, G4).
        # Pin it to False to preserve v7 loss semantics.
        model.accepts_loss_kwargs = False

    # Load tokenizer
    tokenizer_name = config.tokenizer_name or config.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=config.trust_remote_code,
    )

    # Resize embeddings if tokenizer has more tokens than model
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer, hf_config


def apply_lora_with_tied_embeddings(
    model: AutoModelForCausalLM,
    lora_config: LoRAConfig,
    verbose: bool = True,
) -> AutoModelForCausalLM:
    """Apply LoRA with tie_word_embeddings workaround.

    When tie_word_embeddings is True and both lm_head and embed_tokens are in
    modules_to_save, we must:
    1. Remove embed_tokens from modules_to_save (they share weights)
    2. After get_peft_model(), re-tie the weights

    This function isolates the fragile PEFT internals in one place.

    Args:
        model: The base model.
        lora_config: LoRA configuration.
        verbose: Print diagnostic info on rank 0.

    Returns:
        Model with LoRA applied.
    """
    from peft import LoraConfig, get_peft_model

    modules_to_save = list(lora_config.modules_to_save or [])

    # Build PEFT LoRA config
    peft_kwargs = {
        "r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_dropout": lora_config.lora_dropout,
        "target_modules": lora_config.target_modules,
    }

    # Handle tied embeddings
    needs_tie_workaround = False
    if model.config.tie_word_embeddings and modules_to_save:
        has_lm_head = "lm_head" in modules_to_save
        has_embed = "embed_tokens" in modules_to_save
        if has_lm_head and has_embed:
            # Remove embed_tokens — it shares weights with lm_head
            modules_to_save = [m for m in modules_to_save if m != "embed_tokens"]
            needs_tie_workaround = True
            if verbose:
                print(
                    f"[LoRA] tie_word_embeddings=True → "
                    f"modules_to_save adjusted to {modules_to_save}"
                )

    peft_kwargs["modules_to_save"] = modules_to_save or None

    # Handle ensure_weight_tying if available in PEFT
    peft_cfg = LoraConfig(**peft_kwargs)
    if hasattr(peft_cfg, "ensure_weight_tying"):
        peft_cfg.ensure_weight_tying = model.config.tie_word_embeddings

    model = get_peft_model(model, peft_cfg)

    # Re-tie weights after PEFT wrapping (only if PEFT didn't handle it)
    if needs_tie_workaround:
        embed_mod = model.base_model.model.model.embed_tokens
        lm_head_mod = model.base_model.model.lm_head
        # PEFT may have auto-tied via ensure_weight_tying — only fix if needed
        embed_w = embed_mod.weight if not hasattr(embed_mod, "modules_to_save") else embed_mod.modules_to_save["default"].weight
        lm_w = lm_head_mod.modules_to_save["default"].weight if hasattr(lm_head_mod, "modules_to_save") else lm_head_mod.weight
        if embed_w.data_ptr() != lm_w.data_ptr():
            embed_w.data = lm_w.data
            if verbose:
                print("[LoRA] Re-tied embed_tokens ↔ lm_head weights manually")

    if verbose:
        model.print_trainable_parameters()
        # Diagnostic: which devices have trainable params
        devices = {str(p.device) for p in model.parameters() if p.requires_grad}
        print(f"[LoRA] trainable params devices: {devices}")

    return model
