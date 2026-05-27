from tqdm.auto import tqdm
from transformers import AutoTokenizer
import argparse
import os
import json
from pathlib import Path
import copy
import numpy as np
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_tokenizer_path')
    parser.add_argument('--new_tokenizer_path')
    parser.add_argument('--output_path')
    parser.add_argument('--llama3', action='store_true')
    args = parser.parse_args()
    
    with open(Path(args.src_tokenizer_path) / "tokenizer.json") as f:
        old_config = json.load(f)
    
    with open(Path(args.new_tokenizer_path) / "tokenizer.json") as f:
        new_config = json.load(f)
    
    if args.llama3:
        byte_encoder = bytes_to_unicode()
        def token_bytes_to_string(b):
            return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])

        def convert_to_llama(token):
            return token_bytes_to_string(token.replace('▁', ' ').encode())

        def convert_to_llama_merge(merge):
            tokens = merge.split()
            assert len(tokens) == 2 and len(tokens[0]) > 0 and len(tokens[1]) > 0
            return ' '.join([token_bytes_to_string(t.replace('▁', ' ').encode()) for t in tokens])

        new_config_converted = copy.deepcopy(new_config)
        new_config_converted['model']['vocab'] = {}
        new_config_converted['model']['merges'] = []

        for k, v in new_config['model']['vocab'].items():
            k_converted = convert_to_llama(k)
            new_config_converted['model']['vocab'][k_converted] = v

        for m in new_config["model"]["merges"]:
            m_converted = convert_to_llama_merge(m)
            new_config_converted['model']['merges'].append(m_converted)

        new_config = copy.deepcopy(new_config_converted)

    with open(Path(args.src_tokenizer_path) / "tokenizer.json") as f:
        merged_config = json.load(f)
    
    base_size = max(list(old_config["model"]["vocab"].values()))
    cur_idx = base_size + 1
    
    for k in new_config["model"]["vocab"].keys():
        if k not in old_config["model"]["vocab"]:
            merged_config["model"]["vocab"][k] = cur_idx
            cur_idx += 1
    
    print(f"Added {cur_idx - base_size} tokens")
    
    existing_merges = set(new_config["model"]["merges"])
    merged_config["model"]["merges"] = new_config["model"]["merges"]
    for m in old_config["model"]["merges"]:
        if m not in existing_merges:
            merged_config["model"]["merges"].append(m)

    merged_config["model"]["merges"]
    merged_config["model"]["ignore_merges"] = True
    
    
    MERGED_PATH = Path(args.output_path)
    MERGED_PATH.mkdir(exist_ok=True)
    
    with open(MERGED_PATH / "tokenizer.json", mode="w") as f:
        json.dump(merged_config, f, indent=4)
    
    to_copy = ["tokenizer_config.json", "special_tokens_map.json"]
    
    for tc in to_copy:
        with open(Path(args.src_tokenizer_path) / tc) as f:
            tmp = json.load(f)
        with open(MERGED_PATH / tc, mode="w") as f:
            json.dump(tmp, f, indent=4)

    tokenizer = AutoTokenizer.from_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)