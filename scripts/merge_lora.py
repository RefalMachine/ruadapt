import json
import os

import torch
from transformers import AutoTokenizer

from ruadapt.utils.adapter import text_only_adapter_keys
from ruadapt.utils.model_utils import load_causal_lm

def if_lora(model_dir):
    adapter_config_exists = os.path.exists(os.path.join(model_dir, "adapter_config.json"))
    adapter_model_exists = os.path.exists(os.path.join(model_dir, "adapter_model.bin")) or os.path.exists(
        os.path.join(model_dir, "adapter_model.safetensors")
    )
    return adapter_config_exists and adapter_model_exists


def get_input_embeddings(model):
    embeddings = model.get_input_embeddings()
    if embeddings is None:
        raise AttributeError("Model has no input embeddings")
    return embeddings


def get_output_embeddings(model):
    embeddings = model.get_output_embeddings()
    if embeddings is None:
        raise AttributeError("Model has no output embeddings")
    return embeddings


def module_is_saved(modules_to_save, module_name):
    return any(name.rsplit(".", 1)[-1] == module_name for name in (modules_to_save or []))


def get_scaled_weight(module, alpha_scale, scale_embeds):
    if not hasattr(module, "modules_to_save"):
        raise RuntimeError("Expected a PEFT ModulesToSaveWrapper for embeddings")
    original = module.original_module.weight
    trained = module.modules_to_save["default"].weight
    if not scale_embeds or alpha_scale == 1.0:
        return trained.detach().clone()
    return (original + (trained - original) / alpha_scale).detach().clone()


def get_embedding_weights_for_merge(
    input_embeddings, output_embeddings, modules_to_save, tied, alpha_scale, scale_embeds
):
    saved_input = module_is_saved(modules_to_save, "embed_tokens")
    saved_output = module_is_saved(modules_to_save, "lm_head")
    if tied:
        if not saved_input and not saved_output:
            return {}
        source = input_embeddings if hasattr(input_embeddings, "modules_to_save") else output_embeddings
        return {"tied": get_scaled_weight(source, alpha_scale, scale_embeds)}

    weights = {}
    if saved_input:
        weights["input"] = get_scaled_weight(input_embeddings, alpha_scale, scale_embeds)
    if saved_output:
        weights["output"] = get_scaled_weight(output_embeddings, alpha_scale, scale_embeds)
    return weights


def restore_embedding_weights(input_embeddings, output_embeddings, weights):
    with torch.no_grad():
        if "tied" in weights:
            input_embeddings.weight.copy_(weights["tied"])
            output_embeddings.weight = input_embeddings.weight
            return
        if "input" in weights:
            input_embeddings.weight.copy_(weights["input"])
        if "output" in weights:
            output_embeddings.weight.copy_(weights["output"])


def ensure_adapter_is_multimodal(adapter_dir):
    with open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8") as f:
        adapter_config = json.load(f)
    auto_mapping = adapter_config.get("auto_mapping") or {}
    if auto_mapping.get("base_model_class") not in {
        "Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration"
    }:
        return
    unfixed = text_only_adapter_keys(adapter_dir)
    if unfixed:
        base_model = adapter_config.get("base_model_name_or_path", "<multimodal-base-model>")
        raise ValueError(
            f"Adapter still contains {len(unfixed)} text-only Qwen3.5 keys. Fix it before merging:\n"
            f"python -m ruadapt.utils.adapter --input_dir {adapter_dir} --base_model {base_model} --verify"
        )


def merge_lora(model_name: str, output_path: str, device_map: str = "auto", alpha_scale=1.0, scale_embeds=True):
    from peft import PeftConfig, PeftModel

    alpha_scale = float(alpha_scale)
    if alpha_scale == 0:
        raise ValueError("alpha_scale must be non-zero")
    if not isinstance(scale_embeds, bool):
        scale_embeds = str(scale_embeds).lower() == "true"

    if not if_lora(model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        base_model = load_causal_lm(model_name, torch_dtype=torch.bfloat16, device_map=device_map)
        base_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        return

    ensure_adapter_is_multimodal(model_name)
    config = PeftConfig.from_pretrained(model_name)
    if config.alpha_pattern is None:
        config.alpha_pattern = {}
    lm_head_alpha = config.alpha_pattern.get("lm_head", config.lora_alpha)
    config.lora_alpha /= alpha_scale
    for name in config.alpha_pattern:
        config.alpha_pattern[name] /= alpha_scale
    if not scale_embeds and "lm_head" in (config.target_modules or []):
        config.alpha_pattern["lm_head"] = lm_head_alpha

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    base_model = load_causal_lm(
        config.base_model_name_or_path, torch_dtype=torch.bfloat16, device_map=device_map
    )
    lora_model = PeftModel.from_pretrained(
        base_model, model_name, torch_dtype=torch.bfloat16, device_map=device_map, config=config
    )

    tied = bool(getattr(base_model.config, "tie_word_embeddings", False))
    embedding_weights = get_embedding_weights_for_merge(
        get_input_embeddings(lora_model), get_output_embeddings(lora_model), config.modules_to_save,
        tied, alpha_scale, scale_embeds
    )
    lora_model = lora_model.merge_and_unload()
    lora_model.train(False)
    restore_embedding_weights(
        get_input_embeddings(lora_model), get_output_embeddings(lora_model), embedding_weights
    )
    if tied and hasattr(lora_model, "tie_weights"):
        lora_model.tie_weights()

    lora_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

if __name__ == "__main__":
    import fire

    fire.Fire(merge_lora)
