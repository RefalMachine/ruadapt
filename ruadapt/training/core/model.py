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
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5ForConditionalGeneration,
        )
        model_cls = Qwen3_5ForConditionalGeneration

    # Load model
    model = model_cls.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=config.attn_implementation,
        trust_remote_code=config.trust_remote_code,
    )

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
