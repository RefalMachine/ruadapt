#!/usr/bin/env python3
"""
Stage 1: Build Static Token Passport

For each NEW token (present in extended tokenizer but not in base), compute:
- BPE depth and leaf/intermediate classification (in extended tokenizer's merge tree)
- Fragmentation stats: how base tokenizer splits this token into subwords
- Corpus frequency (optional, from darulm_10gb.json when available)

Output: data/token_passport.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ruadapt.tokenization.bpe_tree import build_merge_tree


def load_tokenizer_json(tokenizer_path: str) -> Tuple[Dict[str, int], List[str]]:
    """Load vocab and merges from tokenizer.json."""
    tokenizer_json = Path(tokenizer_path) / "tokenizer.json"
    if not tokenizer_json.exists():
        raise FileNotFoundError(f"tokenizer.json not found at {tokenizer_json}")

    with open(tokenizer_json) as f:
        data = json.load(f)

    vocab = data["model"]["vocab"]  # str -> int
    merges = data["model"]["merges"]  # List[str] "left right"
    return vocab, merges


def find_new_tokens(
    base_vocab: Dict[str, int], ext_vocab: Dict[str, int]
) -> Dict[str, int]:
    """Find tokens present in extended vocab but not in base vocab."""
    base_tokens = set(base_vocab.keys())
    ext_tokens = set(ext_vocab.keys())
    added = ext_tokens - base_tokens
    return {tok: ext_vocab[tok] for tok in added}


def compute_bpe_depths(
    new_token_strs: Dict[str, int],
    tree: Dict[str, Tuple[str, str]],
) -> Tuple[Dict[str, int], Dict[str, str]]:
    """
    Compute BPE depth and token_type for each new token (in extended merge tree).
    """
    depth_cache: Dict[str, int] = {}

    def get_depth(token_str: str) -> int:
        if token_str in depth_cache:
            return depth_cache[token_str]
        if token_str not in tree:
            depth_cache[token_str] = 0
            return 0
        left, right = tree[token_str]
        result = 1 + max(get_depth(left), get_depth(right))
        depth_cache[token_str] = result
        return result

    depths = {}
    types = {}
    for token_str in new_token_strs:
        if token_str in tree:
            types[token_str] = "intermediate"
            depths[token_str] = get_depth(token_str)
        else:
            types[token_str] = "leaf"
            depths[token_str] = 0
    return depths, types


def compute_fragmentation(
    new_tokens: Dict[str, int],
    base_tokenizer_path: str,
    ext_tokenizer_path: str,
) -> Dict[str, Dict]:
    """
    For each new token, decode via extended tokenizer, then encode via base tokenizer
    to get the subword fragmentation that the model would actually use.
    """
    from transformers import AutoTokenizer

    print(f"Loading base tokenizer from {base_tokenizer_path}...")
    base_tok = AutoTokenizer.from_pretrained(base_tokenizer_path)

    print(f"Loading extended tokenizer from {ext_tokenizer_path}...")
    ext_tok = AutoTokenizer.from_pretrained(ext_tokenizer_path)

    print(f"Computing fragmentation for {len(new_tokens)} tokens...")
    result = {}

    for i, (token_str, token_id) in enumerate(new_tokens.items()):
        # Decode token ID to get actual text
        try:
            decoded_text = ext_tok.decode([token_id])
        except Exception:
            decoded_text = token_str

        # Encode decoded text with base tokenizer
        try:
            base_ids = base_tok.encode(decoded_text, add_special_tokens=False)
        except Exception:
            base_ids = [token_id]  # fallback

        result[token_str] = {
            "decoded_text": decoded_text,
            "fragmented_ids": base_ids,
            "num_fragments": len(base_ids),
        }

        if (i + 1) % 5000 == 0:
            print(f"  Processed {i+1}/{len(new_tokens)}")

    return result


def load_corpus_frequencies(
    freq_path: Optional[str], new_token_ids: Set[int]
) -> Dict[int, int]:
    """Load corpus frequency file if available."""
    if freq_path is None or not Path(freq_path).exists():
        return {}

    print(f"Loading corpus frequencies from {freq_path}...")
    with open(freq_path) as f:
        freq_data = json.load(f)

    result = {}
    for k, v in freq_data.items():
        tid = int(k)
        if tid in new_token_ids:
            result[tid] = v
    return result


def classify_corpus_bucket(freq: int) -> str:
    if freq == 0:
        return "dead"
    elif freq < 50:
        return "rare"
    elif freq < 500:
        return "uncommon"
    else:
        return "common"


def main():
    parser = argparse.ArgumentParser(description="Build static token passports")
    parser.add_argument(
        "--base-tokenizer-path",
        type=str,
        default="/workdir/models/Qwen3.5-2B-Base",
        help="Path to base tokenizer",
    )
    parser.add_argument(
        "--ext-tokenizer-path",
        type=str,
        default="/workdir/models/RuadaptQwen3.5-2B-Base-b64-minlen4-mean-v2_tokenizer",
        help="Path to extended tokenizer",
    )
    parser.add_argument(
        "--freq-path",
        type=str,
        default=None,
        help="Path to corpus frequency JSON (optional)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/token_passport.json",
        help="Output path for token passport JSON",
    )

    args = parser.parse_args()

    # Load both tokenizers
    print(f"Loading base tokenizer from {args.base_tokenizer_path}...")
    base_vocab, base_merges = load_tokenizer_json(args.base_tokenizer_path)
    print(f"  Base vocab: {len(base_vocab)} tokens")

    print(f"Loading extended tokenizer from {args.ext_tokenizer_path}...")
    ext_vocab, ext_merges = load_tokenizer_json(args.ext_tokenizer_path)
    print(f"  Extended vocab: {len(ext_vocab)} tokens")

    # Find NEW tokens (in extended but not in base)
    new_tokens = find_new_tokens(base_vocab, ext_vocab)
    print(f"  New tokens (diff): {len(new_tokens)}")

    new_token_ids = set(new_tokens.values())
    print(f"  ID range: {min(new_token_ids)}..{max(new_token_ids)}")

    # Build extended merge tree for BPE depth
    print("Building extended BPE merge tree...")
    ext_tree = build_merge_tree(ext_vocab, ext_merges)
    print(f"  Tree entries: {len(ext_tree)}")

    # Compute BPE depths in extended tree
    print("Computing BPE depths (extended tokenizer)...")
    depths, types = compute_bpe_depths(new_tokens, ext_tree)
    leaf_count = sum(1 for t in types.values() if t == "leaf")
    intermediate_count = sum(1 for t in types.values() if t == "intermediate")
    print(f"  Leaf: {leaf_count}, Intermediate: {intermediate_count}")

    # Compute fragmentation by base tokenizer
    print("Computing base-tokenizer fragmentation...")
    frag_data = compute_fragmentation(new_tokens, args.base_tokenizer_path, args.ext_tokenizer_path)

    # Load corpus frequencies
    corpus_freqs = load_corpus_frequencies(args.freq_path, new_token_ids)
    has_freq = len(corpus_freqs) > 0
    print(f"  Corpus frequencies loaded: {has_freq} ({len(corpus_freqs)} tokens)")

    # Assemble passports
    print("Assembling passports...")
    passports = []
    for token_str in sorted(new_tokens.keys(), key=lambda t: new_tokens[t]):
        token_id = new_tokens[token_str]

        frag = frag_data.get(token_str, {"decoded_text": "", "fragmented_ids": [], "num_fragments": 0})
        freq = corpus_freqs.get(token_id, 0) if has_freq else 0

        passport = {
            "token_id": token_id,
            "token_str": token_str,
            "token_decoded": frag["decoded_text"],
            "bpe_depth": depths.get(token_str, 0),
            "token_type": types.get(token_str, "leaf"),
            "num_base_fragments": frag["num_fragments"],
            "base_fragment_ids": frag["fragmented_ids"],
            "corpus_frequency": freq,
            "is_dead": freq == 0 if has_freq else None,
            "corpus_bucket": classify_corpus_bucket(freq) if has_freq else None,
        }
        passports.append(passport)

    # Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(passports, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(passports)} passports to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Summary
    print("\n=== Summary ===")
    print(f"Total new tokens: {len(passports)}")
    print(f"  Leaf: {leaf_count} ({100*leaf_count/len(passports):.1f}%)")
    print(f"  Intermediate: {intermediate_count} ({100*intermediate_count/len(passports):.1f}%)")

    frag_dist = defaultdict(int)
    for p in passports:
        frag_dist[p["num_base_fragments"]] += 1
    print("  Base-fragmentation distribution:")
    for nfrags in sorted(frag_dist.keys()):
        print(f"    {nfrags} fragments: {frag_dist[nfrags]} tokens")

    depth_dist = defaultdict(int)
    for p in passports:
        depth_dist[p["bpe_depth"]] += 1
    print("  BPE depth distribution (extended tree):")
    for depth in sorted(depth_dist.keys()):
        print(f"    depth={depth}: {depth_dist[depth]} tokens")

    if has_freq:
        bucket_dist = defaultdict(int)
        for p in passports:
            bucket_dist[p["corpus_bucket"]] += 1
        print("  Corpus bucket distribution:")
        for bucket in ["dead", "rare", "uncommon", "common"]:
            if bucket in bucket_dist:
                print(f"    {bucket}: {bucket_dist[bucket]} tokens")


if __name__ == "__main__":
    main()
