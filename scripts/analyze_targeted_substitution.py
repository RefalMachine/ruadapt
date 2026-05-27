"""Analyze targeted substitution: how p_substitute + p_stop affect rare token frequencies.

For each remaining rare token (freq < K), computes expected effective frequency
under targeted substitution with different parameter combinations.

Usage:
    python -m scripts.analyze_targeted_substitution \
        --trim_dir /path/to/trim_output \
        --model_path /path/to/original_model \
        --K 50
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np


def build_child_to_parents(merge_tree: Dict[str, Tuple[str, str]]) -> Dict[str, Set[str]]:
    """Reverse index: child token -> set of parent tokens."""
    c2p: Dict[str, Set[str]] = defaultdict(set)
    for parent, (left, right) in merge_tree.items():
        c2p[left].add(parent)
        c2p[right].add(parent)
    return c2p


def find_ancestors(
    token: str,
    child_to_parents: Dict[str, Set[str]],
    max_depth: int = 15,
) -> List[Tuple[str, int]]:
    """Find all ancestors of token with depth. Returns [(ancestor_str, depth), ...]"""
    ancestors = []
    visited = set()
    queue = [(token, 0)]
    while queue:
        current, depth = queue.pop(0)
        if current in visited or depth > max_depth:
            continue
        visited.add(current)
        for parent in child_to_parents.get(current, set()):
            ancestors.append((parent, depth))
            queue.append((parent, depth + 1))
    return ancestors


def subtree_has_rare(token: str, merge_tree: Dict, rare_set: Set[str], cache: Dict) -> bool:
    """Check if subtree contains any rare token (with memoization)."""
    if token in cache:
        return cache[token]
    if token in rare_set:
        cache[token] = True
        return True
    if token not in merge_tree:
        cache[token] = False
        return False
    left, right = merge_tree[token]
    result = subtree_has_rare(left, merge_tree, rare_set, cache) or \
             subtree_has_rare(right, merge_tree, rare_set, cache)
    cache[token] = result
    return result


def compute_effective_freq(
    rare_str: str,
    rare_freq: int,
    ancestors: List[Tuple[str, int]],
    freq_by_str: Dict[str, int],
    p_substitute: float,
    p_stop: float,
    min_parent_freq: int,
    rare_set: Set[str],
    merge_tree: Dict,
) -> Tuple[float, int, int]:
    """Compute expected effective frequency for a rare token.

    Returns (effective_freq, n_contributing_ancestors, total_ancestor_freq).
    """
    additional = 0.0
    n_contrib = 0
    total_anc_freq = 0

    for anc_str, depth in ancestors:
        anc_freq = freq_by_str.get(anc_str, 0)
        if anc_freq < min_parent_freq:
            continue

        # Probability of reaching this rare token from ancestor:
        # p_substitute * (1-p_stop)^depth
        prob = p_substitute * (1 - p_stop) ** depth
        contribution = prob * anc_freq
        additional += contribution
        n_contrib += 1
        total_anc_freq += anc_freq

    return rare_freq + additional, n_contrib, total_anc_freq


def main():
    parser = argparse.ArgumentParser(description="Analyze targeted substitution parameters")
    parser.add_argument("--trim_dir", type=str, required=True, help="Path to trim_tokenizer output")
    parser.add_argument("--model_path", type=str, required=True, help="Path to original model")
    parser.add_argument("--K", type=int, default=50, help="Frequency threshold")
    parser.add_argument("--min_parent_ratio", type=float, default=3.0,
                        help="Min parent freq as multiple of K (e.g. 3 = freq >= 3*K)")
    parser.add_argument("--max_depth", type=int, default=15, help="Max ancestor search depth")
    parser.add_argument("--freq_file", type=str, default=None,
                        help="Path to trainable token freq JSON (from compute_token_freq.py)")
    args = parser.parse_args()

    # ---- Load trim stats ----
    print("Loading trim stats...")
    with open(os.path.join(args.trim_dir, "trim_stats.json")) as f:
        stats = json.load(f)

    remaining_rare = stats["remaining_rare"]
    print(f"  Remaining rare tokens (freq < {args.K}): {len(remaining_rare)}")

    # ---- Load merge tree ----
    print("Loading merge tree...")
    from transformers import AutoTokenizer
    from ruadapt.training.datasets.utils import _load_merge_tree

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    merge_tree, vocab_str_to_id, safe_tokens = _load_merge_tree(tok)
    id_to_str = {v: k for k, v in vocab_str_to_id.items()}
    print(f"  Merge tree entries: {len(merge_tree)}")

    # ---- Load token frequencies ----
    # Try full freq file first, then trainable-only, then build from remaining_rare
    freq_path_full = os.path.join(args.trim_dir, "token_freq.json")
    freq_path_trainable = os.path.join(args.trim_dir, "trainable_token_freq.json")

    freq_by_str = {}
    freeze_idx = stats.get("freeze_idx", 248044)

    # Priority: --freq_file > token_freq.json > trainable_token_freq.json > remaining_rare
    freq_source = None
    if args.freq_file and os.path.isfile(args.freq_file):
        freq_source = args.freq_file
    elif os.path.isfile(freq_path_full):
        freq_source = freq_path_full
    elif os.path.isfile(freq_path_trainable):
        freq_source = freq_path_trainable

    if freq_source:
        with open(freq_source) as f:
            freq_data = json.load(f)
        raw_freq = freq_data.get("token_freq", freq_data)
        freq_by_str = {id_to_str[int(k)]: v for k, v in raw_freq.items() if int(k) in id_to_str}
        print(f"  Loaded {len(freq_by_str)} token frequencies from {freq_source}")
    else:
        print("  WARNING: No freq file found, using only remaining_rare frequencies")
        for t in remaining_rare:
            if t["token_str"]:
                freq_by_str[t["token_str"]] = t["freq"]

    # For frozen tokens (id < freeze_idx), assume high frequency
    # They're the base vocabulary — always qualify as parents
    frozen_assumed_freq = stats.get("total_tokens", 1_500_000_000) // max(freeze_idx, 1)
    n_frozen_added = 0
    for tid in range(freeze_idx):
        ts = id_to_str.get(tid)
        if ts and ts not in freq_by_str:
            freq_by_str[ts] = frozen_assumed_freq
            n_frozen_added += 1
    if n_frozen_added > 0:
        print(f"  Added {n_frozen_added} frozen tokens with assumed freq={frozen_assumed_freq}")

    # ---- Build structures ----
    print("Building reverse merge tree...")
    child_to_parents = build_child_to_parents(merge_tree)

    rare_set = set()
    rare_tokens = []
    for t in remaining_rare:
        ts = t.get("token_str")
        if ts:
            rare_set.add(ts)
            rare_tokens.append((ts, t["freq"], t.get("id"), t.get("decoded")))
    print(f"  Rare tokens with valid string: {len(rare_tokens)}")

    # ---- Find ancestors for each rare token ----
    print(f"Finding ancestors (max_depth={args.max_depth})...")
    rare_ancestors = {}  # rare_str -> [(ancestor_str, depth), ...]
    for ts, freq, tid, decoded in rare_tokens:
        ancestors = find_ancestors(ts, child_to_parents, max_depth=args.max_depth)
        # Filter: only ancestors with freq >= min_parent_freq
        min_parent_freq = int(args.K * args.min_parent_ratio)
        filtered = [(a, d) for a, d in ancestors if freq_by_str.get(a, 0) >= min_parent_freq]
        rare_ancestors[ts] = filtered

    # Stats on ancestors
    n_with_ancestors = sum(1 for v in rare_ancestors.values() if len(v) > 0)
    n_without = len(rare_ancestors) - n_with_ancestors
    ancestor_counts = [len(v) for v in rare_ancestors.values()]
    print(f"  Tokens with qualifying ancestors: {n_with_ancestors}")
    print(f"  Tokens WITHOUT qualifying ancestors: {n_without}")
    print(f"  Ancestors per token: min={min(ancestor_counts)}, max={max(ancestor_counts)}, "
          f"mean={np.mean(ancestor_counts):.1f}, median={np.median(ancestor_counts):.0f}")

    # ---- Parameter sweep ----
    print("\n" + "=" * 80)
    print("  PARAMETER SWEEP: effective frequency of rare tokens")
    print("=" * 80)

    p_sub_values = [0.1, 0.15, 0.2, 0.25, 0.3]
    p_stop_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    # Header
    print(f"\n{'p_sub':>6} {'p_stop':>6} {'fixed':>6} {'mean':>8} {'median':>8} "
          f"{'p5':>8} {'p25':>8} {'p75':>8} {'p95':>8} {'still<K':>9}")
    print("-" * 80)

    results = []
    for p_sub in p_sub_values:
        for p_stop in p_stop_values:
            effective_freqs = []
            for ts, orig_freq, tid, decoded in rare_tokens:
                ancestors = rare_ancestors.get(ts, [])
                eff, _, _ = compute_effective_freq(
                    ts, orig_freq, ancestors, freq_by_str,
                    p_sub, p_stop, int(args.K * args.min_parent_ratio),
                    rare_set, merge_tree,
                )
                effective_freqs.append(eff)

            eff_arr = np.array(effective_freqs)
            n_fixed = int(np.sum(eff_arr >= args.K))
            still_rare = int(np.sum(eff_arr < args.K))

            row = {
                "p_sub": p_sub,
                "p_stop": p_stop,
                "fixed": n_fixed,
                "mean": float(np.mean(eff_arr)),
                "median": float(np.median(eff_arr)),
                "p5": float(np.percentile(eff_arr, 5)),
                "p25": float(np.percentile(eff_arr, 25)),
                "p75": float(np.percentile(eff_arr, 75)),
                "p95": float(np.percentile(eff_arr, 95)),
                "still_lt_K": still_rare,
            }
            results.append(row)

            print(f"{p_sub:>6.2f} {p_stop:>6.2f} {n_fixed:>6} "
                  f"{row['mean']:>8.1f} {row['median']:>8.1f} "
                  f"{row['p5']:>8.1f} {row['p25']:>8.1f} "
                  f"{row['p75']:>8.1f} {row['p95']:>8.1f} {still_rare:>9}")

    # ---- Top combos ----
    print("\n" + "=" * 80)
    print("  TOP 5 COMBOS (by # fixed tokens)")
    print("=" * 80)
    top = sorted(results, key=lambda x: x["fixed"], reverse=True)[:5]
    print(f"{'p_sub':>6} {'p_stop':>6} {'fixed':>6} {'mean':>8} {'median':>8} {'still<K':>9}")
    print("-" * 50)
    for r in top:
        print(f"{r['p_sub']:>6.2f} {r['p_stop']:>6.2f} {r['fixed']:>6} "
              f"{r['mean']:>8.1f} {r['median']:>8.1f} {r['still_lt_K']:>9}")

    # ---- Save results ----
    output_path = os.path.join(args.trim_dir, "targeted_substitution_analysis.json")
    with open(output_path, "w") as f:
        json.dump({
            "K": args.K,
            "min_parent_ratio": args.min_parent_ratio,
            "max_depth": args.max_depth,
            "n_remaining_rare": len(rare_tokens),
            "n_with_ancestors": n_with_ancestors,
            "n_without_ancestors": n_without,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
