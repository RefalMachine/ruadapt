import os
import json
import random
import argparse
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer
from ruadapt.tokenization.bpe_tree import build_merge_tree, recursive_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_len", type=int, default=4, help="Min character length of decoded token")
    parser.add_argument("--trials", type=int, default=1000, help="Number of split attempts per token")
    parser.add_argument("--min_count", type=int, default=50, help="Min occurrences to keep a split variant")
    args = parser.parse_args()

    model_dir = "/workdir/models/Qwen3.5-2B-Base" # Path to base model (requires external storage due to size)
    tokenizer_file = os.path.join(model_dir, "tokenizer.json")
    
    print("Loading tokenizer data...")
    with open(tokenizer_file, "r", encoding="utf-8") as f:
        tok_data = json.load(f)
        
    model_data = tok_data.get("model", {})
    vocab = model_data.get("vocab", {})
    merges = model_data.get("merges", [])
    
    merge_tree = build_merge_tree(vocab, merges)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    
    print("Building safe token set (filtering out broken utf-8 bytes)...")
    safe_tokens = set()
    for t_str in vocab.keys():
        decoded = tokenizer.convert_tokens_to_string([t_str])
        if '\ufffd' not in decoded:
            safe_tokens.add(t_str)

    print(f"Total vocab: {len(vocab)}. Safe tokens: {len(safe_tokens)}. Merges: {len(merges)}")

    targets = []
    # Identify valid target tokens
    for t_str, t_id in vocab.items():
        if t_str not in merge_tree:
            continue
        decoded = tokenizer.convert_tokens_to_string([t_str])
        if len(decoded) >= args.min_len and '\ufffd' not in decoded:
            targets.append((t_str, t_id, decoded))

    print(f"Found {len(targets)} valid targets of length >= {args.min_len}")
    
    dataset = []
    print("Simulating fragmentation...")
    
    # Process targets
    for t_str, target_id, decoded in tqdm(targets):
        counter = Counter()
        for _ in range(args.trials):
            # p_split=0.5 -> probabilistic descent
            split = recursive_split(t_str, merge_tree, p_split=0.5, force_split=False, safe_tokens=safe_tokens)
            if len(split) > 1: # We only care about actual fragmentation
                # Join by a special separator to hash it in Counter
                counter["||".join(split)] += 1
                
        for split_str, count in counter.items():
            if count >= args.min_count:
                subwords = split_str.split("||")
                # map subwords back to their IDs
                subword_ids = [vocab[sw] for sw in subwords if sw in vocab]
                if len(subword_ids) == len(subwords):
                    dataset.append({
                        "target_id": target_id,
                        "target_str": t_str,
                        "target_decoded": decoded,
                        "fragmented_ids": subword_ids,
                        "fragmented_strs": subwords,
                        "count": count
                    })

    print(f"Total dataset examples (token-split pairs): {len(dataset)}")
    
    # Stratified split: split by unique targets to avoid leaking same token to train and val
    unique_targets = list(set([d["target_id"] for d in dataset]))
    random.shuffle(unique_targets)
    split_idx = int(len(unique_targets) * 0.9)
    train_target_ids = set(unique_targets[:split_idx])
    
    train_data = [d for d in dataset if d["target_id"] in train_target_ids]
    val_data = [d for d in dataset if d["target_id"] not in train_target_ids]
    
    os.makedirs('data', exist_ok=True)
    with open('data/train_bpe.json', 'w') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open('data/val_bpe.json', 'w') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
        
    print(f"Saved dataset. Train: {len(train_data)}, Val: {len(val_data)}")

if __name__ == "__main__":
    main()
