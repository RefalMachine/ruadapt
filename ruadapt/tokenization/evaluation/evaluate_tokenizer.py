import argparse
import os
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import time

def evaluate_on_file(tokenizer, filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        docs = json.load(f)
        
    total_chars = 0
    total_tokens = 0
    
    for doc in docs:
        text = doc["text"]
        total_chars += len(text)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(tokens)
        
    chars_per_token = total_chars / total_tokens if total_tokens > 0 else 0
    return total_chars, total_tokens, chars_per_token

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--data_dir", default="/workdir/projects/tokenizer_extension/evaluation/data")
    args = parser.parse_args()
    
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    
    results = {}
    files = [f for f in os.listdir(args.data_dir) if f.endswith(".json")]
    
    print(f"{'Language':<10} | {'Chars/Token':<12} | {'Tokens':<10} | {'Chars':<10}")
    print("-" * 50)
    
    for file in sorted(files):
        lang = file.split(".")[0]
        filepath = os.path.join(args.data_dir, file)
        
        chars, tokens, cpt = evaluate_on_file(tokenizer, filepath)
        results[lang] = {"chars_per_token": cpt, "tokens": tokens, "chars": chars}
        
        print(f"{lang:<10} | {cpt:<12.4f} | {tokens:<10} | {chars:<10}")

if __name__ == "__main__":
    main()
