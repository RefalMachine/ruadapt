import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftConfig, PeftModel
from typing import List

def scale(config, alpha_scale):
    config.lora_alpha /= alpha_scale
    for name in config.alpha_pattern:
        config.alpha_pattern[name] /= alpha_scale
    return config

def merge_loras(loras_paths: str, loras_ascales: str, output_path: str, device_map: str = "auto"):
    loras_paths = loras_paths.split(',')
    if type(loras_ascales) != list and type(loras_ascales) != tuple:
        loras_ascales = [loras_ascales]

    print(loras_ascales, type(loras_ascales))
    #loras_ascales = list(map(float, loras_ascales.split(',')))
    print(loras_paths)
    
    base_model_path = None
    for lora_path, alpha_scale in zip(*[loras_paths, loras_ascales]):
        config = PeftConfig.from_pretrained(lora_path)
        config = scale(config, alpha_scale)

        if base_model_path is None:
            base_model_path = config.base_model_name_or_path
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                load_in_8bit=False,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
            )
            tie_word_embeddings = model.config.tie_word_embeddings
        else:
            if base_model_path != config.base_model_name_or_path:
                print(base_model_path)
                print(config.base_model_name_or_path)

        model = PeftModel.from_pretrained(
            model, lora_path, torch_dtype=torch.bfloat16, device_map=device_map, config=config
        )

        if tie_word_embeddings and config.modules_to_save is not None and 'lm_head' in config.modules_to_save: 
            print(model.base_model.model.lm_head.original_module.weight[0])
            print(model.base_model.model.lm_head.modules_to_save['default'].weight[0])
            with torch.no_grad():
                delta = model.base_model.model.lm_head.modules_to_save['default'].weight - model.base_model.model.lm_head.original_module.weight
                delta /= alpha_scale
                new_embeds = model.base_model.model.lm_head.original_module.weight + delta
                print(delta.norm())
                
        model = model.merge_and_unload()
        model.train(False)
        if tie_word_embeddings and config.modules_to_save is not None and 'lm_head' in config.modules_to_save:
            with torch.no_grad():
                model.lm_head.weight.copy_(new_embeds)
                model.model.embed_tokens.weight = model.lm_head.weight
        print(model.model.embed_tokens.weight[0])
        print(model.lm_head.weight[0])

    model.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)

if __name__ == "__main__":
    fire.Fire(merge_loras)
