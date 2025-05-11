import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftConfig, PeftModel
import os

def if_lora(model_dir):
    adapter_config_exists = os.path.exists(os.path.join(model_dir, 'adapter_config.json'))
    adapter_model_exists = os.path.exists(os.path.join(model_dir, 'adapter_model.bin')) or os.path.exists(os.path.join(model_dir, 'adapter_model.safetensors'))
    return adapter_config_exists and adapter_model_exists

def merge_lora(model_name: str, output_path: str, device_map: str = "auto", alpha_scale=1.0, scale_embeds=True):
    scale_embeds = scale_embeds=='true'
    print(model_name, output_path, device_map, alpha_scale, scale_embeds, scale_embeds==True)
    if not if_lora(model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=False,
            torch_dtype=torch.bfloat16,
            device_map='cpu',
        )
        print(base_model)
        base_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        return 

    print(model_name)
    print(output_path)
    print(alpha_scale)
    config = PeftConfig.from_pretrained(model_name)
    lm_head_alpha = config.alpha_pattern.get("lm_head", config.lora_alpha)

    config.lora_alpha /= alpha_scale
    for name in config.alpha_pattern:
        config.alpha_pattern[name] /= alpha_scale

    if not scale_embeds and 'lm_head' in config.target_modules:
        config.alpha_pattern["lm_head"] = lm_head_alpha
    #if True:
    #    config.alpha_pattern["lm_head"] = lm_head_alpha

    #config.lora_alpha /= alpha_scale
    print(config.lora_alpha)
    base_model_path = config.base_model_name_or_path
    #generation_config = GenerationConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        device_map='cpu',
    )

    print(config)
    lora_model = PeftModel.from_pretrained(
        base_model, model_name, torch_dtype=torch.bfloat16, device_map='cpu', config=config
    )
    
    if config.modules_to_save is not None and 'lm_head' in config.modules_to_save: 
        print(lora_model.base_model.model.lm_head.original_module.weight[0])
        print(lora_model.base_model.model.lm_head.modules_to_save['default'].weight[0])
        with torch.no_grad():
            delta = lora_model.base_model.model.lm_head.modules_to_save['default'].weight - lora_model.base_model.model.lm_head.original_module.weight
            delta /= alpha_scale if scale_embeds else 1.0
            new_embeds = lora_model.base_model.model.lm_head.original_module.weight + delta

    lora_model = lora_model.merge_and_unload()
    lora_model.train(False)

    #lora_model.generation_config = generation_config
    print(lora_model.model.embed_tokens.weight[0])
    print(lora_model.lm_head.weight[0])
    print(base_model.config.tie_word_embeddings)
    print(config.modules_to_save)
    if config.modules_to_save is not None and 'lm_head' in config.modules_to_save:
        with torch.no_grad():
            lora_model.lm_head.weight.copy_(new_embeds)
            if base_model.config.tie_word_embeddings:
                lora_model.model.embed_tokens.weight = lora_model.lm_head.weight

    print(lora_model.model.embed_tokens.weight[0])
    print(lora_model.lm_head.weight[0])

    lora_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    #generation_config.save_pretrained(output_path)

if __name__ == "__main__":
    fire.Fire(merge_lora)
