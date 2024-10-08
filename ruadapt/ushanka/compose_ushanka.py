import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GenerationConfig, LogitsProcessorList
import argparse
import time
import json
from pathlib import Path

from src.ushanka import make_ushanka
from src.ushanka_proj_utils import list_projection_modes

def load_model(model_path):
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=config.torch_dtype,
        device_map="cuda:0",
        attn_implementation="sdpa",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Model loaded")
    time.sleep(2)
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--mode", type=str, default="conversion")
    parser.add_argument("--out_path", type=str, default="./composed_models")
    args = parser.parse_args()
    
    out_dir = Path(args.out_path)
    out_dir.mkdir(exist_ok=True)
    
    config_path = Path(args.config_path)
    out_name = f"{config_path.stem}_{args.mode}"
    print(f"Output will placed at: {out_dir / out_name}")
    
    with open(config_path) as f:
        config_dict = json.load(f)
    print("Will be composed from",
          *[f"{k.upper()}: {v}" for k,v in config_dict.items()],
          sep='\n'
         )
    
    target_model, target_model_tokenizer = load_model(config_dict["target_model_path"])
    source_model, source_model_tokenizer = load_model(config_dict["source_model_path"])
    donor_model, donor_model_tokenizer = load_model(config_dict["donor_model_path"])
    
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
    model.config.vocab_size = len(tokenizer.get_vocab())
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    
    model.save_pretrained(out_dir / out_name)
    tokenizer.save_pretrained(out_dir / out_name)