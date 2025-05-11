import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GenerationConfig, LogitsProcessorList
import argparse
import time
import json
import codecs
from pathlib import Path
from huggingface_hub import snapshot_download
from .src.ushanka import make_ushanka
from .src.ushanka_proj_utils import list_projection_modes
import shutil
import os
from safetensors import safe_open
from safetensors.torch import save_file
import gc

def check_if_lora(model_dir):
    if_lora = False
    if os.path.exists(model_dir):
        adapter_config_exists = os.path.exists(os.path.join(model_dir, 'adapter_config.json'))
        adapter_model_exists = os.path.exists(os.path.join(model_dir, 'adapter_model.bin')) or os.path.exists(os.path.join(model_dir, 'adapter_model.safetensors'))
        if_lora = adapter_config_exists and adapter_model_exists
        return if_lora
    try:
        PeftConfig.from_pretrained(model_dir)
        if_lora  = True
    except:
        pass
    return if_lora

def load_model(model_path, device="cpu"):
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=config.torch_dtype,
        device_map=device,
        attn_implementation="sdpa",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Model loaded")
    time.sleep(2)
    return model, tokenizer

def load_prune_save_lora(model_path, key_filter):
    tensors = {}
    with safe_open(Path(model_path)/"adapter_model.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            if key_filter(key):
                tensors[key] = f.get_tensor(key)
                
    with codecs.open(Path(model_path)/'adapter_config.json', 'r', 'utf-8') as config_file:
        config = json.load(config_file)
        
    if config['modules_to_save'] is not None:
        config['modules_to_save'] = [k for k in config['modules_to_save'] if key_filter(k)]

    config['target_modules'] = [k for k in config['target_modules'] if key_filter(k)]

    if len(config['target_modules']) == 0:
        config['target_modules'] = 'dummy-target-modules'
        
    if config['modules_to_save']  is None or len(config['modules_to_save']) == 0:
        config['modules_to_save'] = None
        
    assert config['modules_to_save'] is not None or config['target_modules']

    with codecs.open(Path(model_path)/'adapter_config.json', 'w', 'utf-8') as config_file:
        json.dump(config, config_file, indent=4)
        
    save_file(tensors, Path(model_path)/'adapter_model.safetensors')

def save_split_lora(model_path, out_dir):
    if not os.path.exists(model_path):
        snapshot_download(model_path, local_dir=out_dir/'full')
        model_path = out_dir/'full'
    

    if os.path.exists(out_dir/'embeds'):
        shutil.rmtree(out_dir/'embeds')
    shutil.copytree(model_path, out_dir/'embeds')

    if os.path.exists(out_dir/'adapters'):
        shutil.rmtree(out_dir/'adapters')
    shutil.copytree(model_path, out_dir/'adapters')


    load_prune_save_lora(out_dir/'embeds', lambda key: 'lm_head' in key or 'embed_tokens' in key)
    load_prune_save_lora(out_dir/'adapters', lambda key: 'lm_head' not in key and 'embed_tokens' not in key)

    return out_dir/'embeds', out_dir/'adapters'

def load_lora(model_path, base_model_path=None, alpha_scale=1.0, not_scale_lm_head=False, device='cuda:0'):
    config = PeftConfig.from_pretrained(model_path)
    lm_head_alpha = config.alpha_pattern.get("lm_head", config.lora_alpha)

    config.lora_alpha /= alpha_scale
    for name in config.alpha_pattern:
        config.alpha_pattern[name] /= alpha_scale

    if not_scale_lm_head:
        config.alpha_pattern["lm_head"] = lm_head_alpha

    base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
    torch_dtype = base_model_config.torch_dtype
    base_model_path = config.base_model_name_or_path if base_model_path is None else base_model_path
    print(config)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map=device,
        attn_implementation="sdpa"
    )

    model = PeftModel.from_pretrained(
        model,
        model_path,
        torch_dtype=torch_dtype,
        config=config
    )

    if model.config.tie_word_embeddings and config.modules_to_save is not None and 'lm_head' in config.modules_to_save: 
        with torch.no_grad():
            delta = model.base_model.model.lm_head.modules_to_save['default'].weight - model.base_model.model.lm_head.original_module.weight
            delta /= alpha_scale
            new_embeds = model.base_model.model.lm_head.original_module.weight + delta

    model = model.merge_and_unload()
    model.train(False)

    print(model.model.embed_tokens.weight[0])
    print(model.lm_head.weight[0])
    print(model.config.tie_word_embeddings)
    print(config.modules_to_save)

    if model.config.tie_word_embeddings and config.modules_to_save is not None and 'lm_head' in config.modules_to_save:
        assert 'lm_head' not in config.modules_to_save or 'embed_tokens' not in config.modules_to_save
        with torch.no_grad():
            model.lm_head.weight.copy_(new_embeds)
            model.model.embed_tokens.weight = model.lm_head.weight

    print(model.model.embed_tokens.weight[0])
    print(model.lm_head.weight[0])

    model.train(False)
    return model

def load_donor_model(model_path, out_dir, alpha_scale=1.0, not_scale_lm_head=False, device="cuda:0"):
    adapters_path = None
    if check_if_lora(model_path):
        embeds_path, adapters_path = save_split_lora(model_path, out_dir)
        model_path = embeds_path
        model = load_lora(model_path, None, alpha_scale, not_scale_lm_head, device=device)
    else:
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=config.torch_dtype,
            device_map=device,
            attn_implementation="sdpa",
        )
        
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    print("Model loaded")
    time.sleep(2)

    return model, tokenizer, adapters_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--mode", type=str, default="conversion")
    parser.add_argument("--output_path", type=str, default="./composed_models")
    parser.add_argument("--custom_chat_template_path", type=str, default=None)
    args = parser.parse_args()
    
    print(f'LEP MODE USHANKA {args.mode}')

    out_dir = Path(args.output_path)
    out_dir.mkdir(exist_ok=True)
    
    config_path = Path(args.config_path)
    print(f"Output will placed at: {out_dir}")
    #print(f"Output will placed at: {out_dir / out_name}")
    
    with open(config_path) as f:
        config_dict = json.load(f)
    print("Will be composed from",
          *[f"{k.upper()}: {v}" for k,v in config_dict.items()],
          sep='\n'
         )
    
    target_model, target_model_tokenizer = load_model(config_dict["target_model_path"], device="cuda:0")
    source_model, source_model_tokenizer = load_model(config_dict["source_model_path"], device="cuda:1")
    donor_model, donor_model_tokenizer, adapters_path = load_donor_model(
        config_dict["donor_model_path"], out_dir, config_dict.get('alpha_scale', 1.0), config_dict.get('not_scale_lm_head', False),
        device='cuda:2'
    )
    
    proj_modes = {"lm_head": args.mode,
              "model.embed_tokens": args.mode}

    model, tokenizer = make_ushanka(
        target_model, source_model, donor_model,
        target_model_tokenizer, donor_model_tokenizer, module_projection_modes=proj_modes,
        coocurrence_map_path=None,
        overlap_penalty=1.0)
    
    # Fix bad configs
    model.generation_config.do_sample = True
    
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = len(tokenizer.get_vocab())
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    model.save_pretrained(out_dir)
    if args.custom_chat_template_path is not None:
        with codecs.open(args.custom_chat_template_path, 'r', 'utf-8') as file:
            tokenizer.chat_template = json.load(file)

    tokenizer.save_pretrained(out_dir)

    
    del model
    gc.collect()
    torch.cuda.empty_cache()

    if adapters_path is not None:
        model = load_lora(adapters_path, out_dir, config_dict.get('alpha_scale', 1.0), config_dict.get('not_scale_lm_head', False), device='cuda:3')
        model.save_pretrained(out_dir)
