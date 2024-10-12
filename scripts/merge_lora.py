import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftConfig, PeftModel


def merge_lora(model_name: str, output_path: str, device_map: str = "auto"):
    config = PeftConfig.from_pretrained(model_name)
    base_model_path = config.base_model_name_or_path
    #generation_config = GenerationConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    lora_model = PeftModel.from_pretrained(
        base_model, model_name, torch_dtype=torch.bfloat16, device_map=device_map
    )

    lora_model = lora_model.merge_and_unload()
    lora_model.train(False)

    #lora_model.generation_config = generation_config
    print(lora_model.model.embed_tokens.weight[0])
    print(lora_model.lm_head.weight[0])
    print(base_model.config.tie_word_embeddings)
    print(config.modules_to_save)
    if base_model.config.tie_word_embeddings and config.modules_to_save is not None and 'lm_head' in config.modules_to_save:
        lora_model.model.embed_tokens.weight = lora_model.lm_head.weight

    print(lora_model.model.embed_tokens.weight[0])
    print(lora_model.lm_head.weight[0])

    lora_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    #generation_config.save_pretrained(output_path)



if __name__ == "__main__":
    fire.Fire(merge_lora)
