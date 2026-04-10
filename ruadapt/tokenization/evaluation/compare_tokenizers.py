import argparse
import os
import json
from transformers import AutoTokenizer

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
        
    return total_chars, total_tokens

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(script_dir, "data")

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_tokenizer", default="/workdir/models/Qwen3.5-2B-Base")
    parser.add_argument("--new_tokenizer", default="/workdir/models/Qwen3.5-2B-Base/hf_tokenizer_u48_min4")
    parser.add_argument("--data_dir", default=default_data_dir)
    args = parser.parse_args()
    
    print("Loading base tokenizer...")
    tokenizer_base = AutoTokenizer.from_pretrained(args.base_tokenizer, trust_remote_code=True)
    print("Loading new tokenizer...")
    tokenizer_new = AutoTokenizer.from_pretrained(args.new_tokenizer, trust_remote_code=True)
    
    files = [f for f in os.listdir(args.data_dir) if f.endswith(".json")]
    
    print(f"\n{'Language':<10} | {'Base CPT':<10} | {'New CPT':<10} | {'Diff %':<8} | {'Status'}")
    print("-" * 65)
    
    for file in sorted(files):
        lang = file.split(".")[0]
        filepath = os.path.join(args.data_dir, file)
        
        chars, base_toks = evaluate_on_file(tokenizer_base, filepath)
        _, new_toks = evaluate_on_file(tokenizer_new, filepath)
        
        base_cpt = chars / base_toks if base_toks > 0 else 0
        new_cpt = chars / new_toks if new_toks > 0 else 0
        
        diff_pct = ((new_cpt - base_cpt) / base_cpt) * 100 if base_cpt > 0 else 0
        
        # If tokens changed, but it's not Russian (or Ukrainian/Kazakh which share cyrillic), flag it
        status = "OK"
        if base_toks != new_toks:
            if lang in ["rus", "ukr", "kaz"]:
                status = "EXPECTED (Cyrillic)"
            else:
                status = f"WARNING! {base_toks} -> {new_toks}"
                
        print(f"{lang:<10} | {base_cpt:<10.4f} | {new_cpt:<10.4f} | {diff_pct:>+6.2f}% | {status}")

if __name__ == "__main__":
    main()
