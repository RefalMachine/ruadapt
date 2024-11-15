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
    print(loras_ascales)
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
            model = PeftModel.from_pretrained(
                model, lora_path, torch_dtype=torch.bfloat16, device_map=device_map, config=config
            )
            model = model.merge_and_unload()
            model.train(False)
        else:
            if base_model_path != config.base_model_name_or_path:
                print(base_model_path)
                print(config.base_model_name_or_path)
            #assert base_model_path == config.base_model_name_or_path
            model = PeftModel.from_pretrained(
                model, lora_path, torch_dtype=torch.bfloat16, device_map=device_map, config=config
            )
            model = model.merge_and_unload()
            model.train(False)

        if model.config.tie_word_embeddings and config.modules_to_save is not None and 'lm_head' in config.modules_to_save:
            model.model.embed_tokens.weight = model.lm_head.weight

    model.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)

if __name__ == "__main__":
    fire.Fire(merge_loras)
