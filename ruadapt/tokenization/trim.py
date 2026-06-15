"""Cascade-trim rare terminal tokens from a tokenizer vocabulary.

Algorithm:
  1. Tokenize dataset, count token frequencies.
  2. Classify trainable tokens as leaf / terminal / intermediate via BPE merge tree.
  3. Cascade: remove terminal tokens with freq < K, pour their frequencies into
     their subword decomposition. Recompute newly-terminal tokens. Repeat.
  4. Save trimmed tokenizer and statistics JSON.

CLI entrypoint:
    python -m ruadapt.tokenization.trim \
        --model_path /path/to/model \
        --data_path /path/to/train.jsonl \
        --output_dir /path/to/output \
        --freeze_idx 248044 \
        --K 50 \
        --num_proc 16
"""

import argparse
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer

from ruadapt.tokenization.bpe_tree import build_merge_tree, recursive_split
from ruadapt.tokenization.utils import get_special_token_ids
from ruadapt.training.datasets.utils import _get_tokenize_fn, _load_merge_tree, _log


# ---------------------------------------------------------------------------
# Token classification
# ---------------------------------------------------------------------------

def classify_tokens(
    vocab_str_to_id: Dict[str, int],
    merge_tree: Dict[str, Tuple[str, str]],
    trainable_ids: Set[int],
    freeze_idx: int,
    eos_id: int,
) -> Tuple[Dict[str, str], Dict[str, Set[str]]]:
    """Classify trainable tokens as leaf / terminal / intermediate.

    Also builds a reverse map: child_token_str -> set of parent_token_str
    (i.e. parent is in merge_tree and its (left, right) contains child).

    Returns:
        token_class: token_str -> 'leaf' | 'terminal' | 'intermediate'
        child_to_parents: token_str -> set of parent token_strs
    """
    child_to_parents: Dict[str, Set[str]] = defaultdict(set)
    for parent, (left, right) in merge_tree.items():
        child_to_parents[left].add(parent)
        child_to_parents[right].add(parent)

    id_to_str = {v: k for k, v in vocab_str_to_id.items()}

    token_class: Dict[str, str] = {}
    for tid in trainable_ids:
        ts = id_to_str.get(tid)
        if ts is None:
            continue

        in_merge_tree = ts in merge_tree
        has_parents = ts in child_to_parents and len(child_to_parents[ts]) > 0

        if not in_merge_tree:
            token_class[ts] = 'leaf'
        elif not has_parents:
            token_class[ts] = 'terminal'
        else:
            token_class[ts] = 'intermediate'

    return token_class, child_to_parents


# ---------------------------------------------------------------------------
# Cascade removal with frequency pouring
# ---------------------------------------------------------------------------

def cascade_remove(
    token_freq: Dict[int, int],
    token_class: Dict[str, str],
    child_to_parents: Dict[str, Set[str]],
    merge_tree: Dict[str, Tuple[str, str]],
    vocab_str_to_id: Dict[str, int],
    id_to_str: Dict[int, str],
    safe_tokens: Set[str],
    freeze_idx: int,
    special_ids: Set[int],
    K: int,
) -> Tuple[Set[int], Set[str], List[Dict]]:
    """Cascade removal of rare terminal tokens with frequency pouring.

    When a terminal token T (freq < K) is removed:
      - T splits into subwords [A, B, ...] via merge tree
      - freq[T] is poured into freq[A], freq[B], ...
      - If an intermediate token Y loses ALL non-frozen parents, it becomes terminal

    Returns:
        removed_ids: set of removed token IDs
        removed_strs: set of removed token strings
        iterations: list of per-iteration stats dicts
    """
    removed_ids: Set[int] = set()
    removed_strs: Set[str] = set()
    iterations: List[Dict] = []

    freq_by_str: Dict[str, int] = {}
    for tid, cnt in token_freq.items():
        ts = id_to_str.get(tid)
        if ts is not None:
            freq_by_str[ts] = cnt

    alive_parents: Dict[str, Set[str]] = {}
    for child, parents in child_to_parents.items():
        alive_parents[child] = set(parents)

    iteration = 0
    while True:
        iteration += 1

        to_remove_strs: Set[str] = set()
        for ts, cls in token_class.items():
            if cls == 'terminal' and ts not in removed_strs:
                freq = freq_by_str.get(ts, 0)
                if freq < K:
                    tid = vocab_str_to_id.get(ts)
                    if tid is not None and tid >= freeze_idx and tid not in special_ids:
                        to_remove_strs.add(ts)

        if not to_remove_strs:
            break

        iter_removed_ids: List[int] = []
        iter_poured: int = 0
        for ts in to_remove_strs:
            tid = vocab_str_to_id[ts]
            removed_ids.add(tid)
            removed_strs.add(ts)
            iter_removed_ids.append(tid)

            freq = freq_by_str.get(ts, 0)
            if freq > 0 and ts in merge_tree:
                left, right = merge_tree[ts]
                subwords = recursive_split(ts, merge_tree, p_split=1.0, force_split=True, safe_tokens=safe_tokens)
                if len(subwords) > 1:
                    per_sub = freq // len(subwords)
                    remainder = freq % len(subwords)
                    for i, sw in enumerate(subwords):
                        add = per_sub + (1 if i < remainder else 0)
                        freq_by_str[sw] = freq_by_str.get(sw, 0) + add
                    iter_poured += freq

            token_class[ts] = 'removed'

            if ts in merge_tree:
                left, right = merge_tree[ts]
                for child in [left, right]:
                    if child in alive_parents:
                        alive_parents[child].discard(ts)

        newly_terminal: List[str] = []
        for ts, cls in token_class.items():
            if cls == 'intermediate':
                alive = alive_parents.get(ts, set())
                has_frozen_parent = False
                has_alive_parent = False
                for parent_ts in alive:
                    parent_tid = vocab_str_to_id.get(parent_ts)
                    if parent_tid is not None and parent_tid < freeze_idx:
                        has_frozen_parent = True
                    else:
                        has_alive_parent = True

                if has_frozen_parent:
                    continue
                elif not has_alive_parent:
                    token_class[ts] = 'terminal'
                    newly_terminal.append(ts)

        iter_info = {
            "iteration": iteration,
            "removed_count": len(iter_removed_ids),
            "removed_ids_sample": iter_removed_ids[:20],
            "freq_poured": iter_poured,
            "newly_terminal": len(newly_terminal),
            "total_removed": len(removed_ids),
        }
        iterations.append(iter_info)
        print(f"  Iter {iteration}: removed {len(iter_removed_ids)} tokens, "
              f"poured freq {iter_poured:,}, "
              f"{len(newly_terminal)} newly terminal, "
              f"total removed: {len(removed_ids)}")

    return removed_ids, removed_strs, iterations


# ---------------------------------------------------------------------------
# Tokenizer rebuilding
# ---------------------------------------------------------------------------

def rebuild_tokenizer(
    model_path: str,
    removed_strs: Set[str],
    output_dir: str,
) -> Tuple[Dict[int, int], Dict]:
    """Rebuild tokenizer after removing tokens, reindex IDs, align to 256.

    Args:
        model_path: Path to the original model/tokenizer directory.
        removed_strs: Set of token strings to remove.
        output_dir: Directory to save trimmed tokenizer files.

    Returns:
        old_to_new: Mapping from old token IDs to new token IDs.
        tok_data: The modified tokenizer.json data dict.
    """
    tok_json_path = os.path.join(model_path, "tokenizer.json")
    with open(tok_json_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)

    model_data = tok_data.get("model", {})
    original_vocab = dict(model_data.get("vocab", {}))
    original_merges = list(model_data.get("merges", []))

    new_vocab: Dict[str, int] = {}
    old_to_new: Dict[int, int] = {}
    new_id = 0
    for token_str, old_id in sorted(original_vocab.items(), key=lambda x: x[1]):
        if token_str not in removed_strs:
            new_vocab[token_str] = new_id
            old_to_new[old_id] = new_id
            new_id += 1

    new_merges = []
    for merge_rule in original_merges:
        if isinstance(merge_rule, list):
            if len(merge_rule) == 2:
                left, right = merge_rule
            else:
                continue
        else:
            parts = merge_rule.split(" ")
            if len(parts) == 2:
                left, right = parts
            else:
                continue

        if left not in removed_strs and right not in removed_strs and (left + right) not in removed_strs:
            new_merges.append(merge_rule)

    tok_data["model"]["vocab"] = new_vocab
    tok_data["model"]["merges"] = new_merges

    next_id = len(new_vocab)
    special_old_to_new: Dict[int, int] = {}
    if "added_tokens" in tok_data:
        new_added = []
        for entry in tok_data["added_tokens"]:
            old_id = entry.get("id")
            special_old_to_new[old_id] = next_id
            entry["id"] = next_id
            new_added.append(entry)
            next_id += 1
        tok_data["added_tokens"] = new_added

    # Align total vocab size to multiple of 256
    total_vocab = len(new_vocab) + len(tok_data.get("added_tokens", []))
    remainder = total_vocab % 256
    if remainder != 0:
        added = tok_data["added_tokens"]
        free_tokens = [e for e in added if "free_token" in e.get("content", "")]
        real_special = [e for e in added if "free_token" not in e.get("content", "")]

        if remainder <= len(free_tokens):
            removed_free = free_tokens[len(free_tokens) - remainder:]
            free_tokens = free_tokens[:len(free_tokens) - remainder]
            for e in removed_free:
                special_old_to_new.pop(e["id"], None)
            print(f"  Removed {remainder} free_tokens to align vocab to 256")
        else:
            to_add = 256 - remainder
            max_n = 0
            for e in free_tokens:
                c = e.get("content", "")
                try:
                    n = int(c.split("free_token")[1].rstrip("|>"))
                    max_n = max(max_n, n)
                except (IndexError, ValueError):
                    pass
            for i in range(1, to_add + 1):
                free_tokens.append({
                    "content": f"<|free_token{max_n + i}|>",
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                })
            print(f"  Added {to_add} free_tokens to align vocab to 256")

        added = real_special + free_tokens
        new_special_old_to_new: Dict[int, int] = {}
        next_id = len(new_vocab)
        for entry in added:
            old_id = entry["id"]
            new_special_old_to_new[old_id] = next_id
            entry["id"] = next_id
            next_id += 1
        tok_data["added_tokens"] = added

        final_special_old_to_new: Dict[int, int] = {}
        for orig_old_id, intermediate_id in special_old_to_new.items():
            if intermediate_id in new_special_old_to_new:
                final_special_old_to_new[orig_old_id] = new_special_old_to_new[intermediate_id]
        special_old_to_new = final_special_old_to_new

        total_vocab = len(new_vocab) + len(added)
        print(f"  Total vocab size: {total_vocab} (multiple of 256: {total_vocab % 256 == 0})")

    old_to_new.update(special_old_to_new)

    # Save trimmed tokenizer.json
    trimmed_tok_path = os.path.join(output_dir, "tokenizer.json")
    with open(trimmed_tok_path, "w", encoding="utf-8") as f:
        json.dump(tok_data, f, ensure_ascii=False, indent=2)
    print(f"  Saved trimmed tokenizer to {trimmed_tok_path}")

    # Copy and update tokenizer_config.json
    tok_config_src = os.path.join(model_path, "tokenizer_config.json")
    if os.path.isfile(tok_config_src):
        with open(tok_config_src) as f:
            tok_config = json.load(f)

        id_fields = ["bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id",
                      "cls_token_id", "sep_token_id", "mask_token_id"]
        for field in id_fields:
            if field in tok_config and tok_config[field] in special_old_to_new:
                tok_config[field] = special_old_to_new[tok_config[field]]

        tok_config_dst = os.path.join(output_dir, "tokenizer_config.json")
        with open(tok_config_dst, "w", encoding="utf-8") as f:
            json.dump(tok_config, f, ensure_ascii=False, indent=2)
        print(f"  Saved tokenizer_config to {tok_config_dst}")

    special_map_src = os.path.join(model_path, "special_tokens_map.json")
    if os.path.isfile(special_map_src):
        shutil.copy2(special_map_src, os.path.join(output_dir, "special_tokens_map.json"))
        print(f"  Copied special_tokens_map.json")

    return old_to_new, tok_data


# ---------------------------------------------------------------------------
# Statistics & output
# ---------------------------------------------------------------------------

def save_trim_outputs(
    output_dir: str,
    tokenizer,
    old_to_new: Dict[int, int],
    removed_ids: Set[int],
    removed_strs: Set[str],
    id_to_str: Dict[int, str],
    trainable_freq: Dict[int, int],
    class_counts: Counter,
    iterations: List[Dict],
    vocab_size: int,
    new_vocab_size: int,
    freeze_idx: int,
    trainable_ids: Set[int],
    K: int,
    total_tokens: int,
    n_docs: int,
    model_path: str,
    data_path: str,
    freq: Counter,
) -> None:
    """Save mapping, stats, and token frequency files."""
    remaining_trainable = set(old_to_new[tid] for tid in trainable_ids if tid not in removed_ids)

    new_freq: Dict[int, int] = {}
    for old_tid, cnt in trainable_freq.items():
        if old_tid in old_to_new:
            new_tid = old_to_new[old_tid]
            new_freq[new_tid] = cnt

    rare_after = sum(1 for v in new_freq.values() if 0 < v < K)
    zero_after = sum(1 for v in new_freq.values() if v == 0)
    rare_before = sum(1 for v in trainable_freq.values() if 0 < v < K)
    zero_before = sum(1 for v in trainable_freq.values() if v == 0)

    removed_tokens_info = []
    for tid in sorted(removed_ids):
        ts = id_to_str.get(tid)
        decoded = None
        if ts is not None:
            try:
                decoded = tokenizer.convert_tokens_to_string([ts])
            except Exception:
                decoded = None
        removed_tokens_info.append({
            "id": tid,
            "token_str": ts,
            "decoded": decoded,
            "freq": trainable_freq.get(tid, 0),
        })

    mapping_path = os.path.join(output_dir, "token_id_mapping.json")
    mapping_data = {
        "old_vocab_size": vocab_size,
        "new_vocab_size": new_vocab_size,
        "freeze_idx": freeze_idx,
        "removed_count": len(removed_ids),
        "removed_ids": sorted(removed_ids),
        "removed_tokens": removed_tokens_info,
        "old_to_new": {str(k): v for k, v in old_to_new.items()},
    }
    with open(mapping_path, "w") as f:
        json.dump(mapping_data, f, ensure_ascii=False)
    print(f"  Saved ID mapping to {mapping_path}")

    stats = {
        "model_path": model_path,
        "data_path": data_path,
        "n_docs": n_docs,
        "total_tokens": total_tokens,
        "K": K,
        "freeze_idx": freeze_idx,
        "old_vocab_size": vocab_size,
        "new_vocab_size": new_vocab_size,
        "removed_count": len(removed_ids),
        "removed_ids": sorted(removed_ids),
        "class_counts_before": dict(class_counts),
        "rare_before": rare_before,
        "rare_after": rare_after,
        "zero_before": zero_before,
        "zero_after": zero_after,
        "cascade_iterations": iterations,
        "trainable_before": len(trainable_ids),
        "trainable_after": len(remaining_trainable),
        "remaining_rare": [
            {
                "id": old_tid,
                "new_id": old_to_new[old_tid],
                "token_str": id_to_str.get(old_tid),
                "decoded": tokenizer.convert_tokens_to_string([id_to_str.get(old_tid)]) if id_to_str.get(old_tid) else None,
                "freq": cnt,
            }
            for old_tid, cnt in sorted(trainable_freq.items(), key=lambda x: x[1])
            if 0 <= cnt < K and old_tid not in removed_ids
        ],
    }

    stats_path = os.path.join(output_dir, "trim_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved stats to {stats_path}")

    freq_path = os.path.join(output_dir, "token_freq.json")
    with open(freq_path, "w") as f:
        json.dump({"token_freq": {str(k): v for k, v in freq.most_common()}}, f)
    print(f"  Saved token frequencies to {freq_path}")


def print_summary(
    vocab_size: int,
    new_vocab_size: int,
    removed_ids: Set[int],
    iterations: List[Dict],
    trainable_ids: Set[int],
    remaining_trainable: Set[int],
    K: int,
    trainable_freq: Dict[int, int],
    new_freq: Dict[int, int],
) -> None:
    """Print trim summary to stdout."""
    rare_before = sum(1 for v in trainable_freq.values() if 0 < v < K)
    rare_after = sum(1 for v in new_freq.values() if 0 < v < K)

    print("\n" + "=" * 60)
    print("  TRIM SUMMARY")
    print("=" * 60)
    print(f"  Old vocab size:       {vocab_size:,}")
    print(f"  New vocab size:       {new_vocab_size:,}")
    print(f"  Removed tokens:       {len(removed_ids):,}")
    print(f"  Cascade iterations:   {len(iterations)}")
    print(f"  Trainable before:     {len(trainable_ids):,}")
    print(f"  Trainable after:      {len(remaining_trainable):,}")
    print(f"  Rare (0 < f < {K}) before: {rare_before}")
    print(f"  Rare (0 < f < {K}) after:  {rare_after}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def trim_tokenizer(
    model_path: str,
    data_path: str,
    output_dir: str,
    freeze_idx: int = 248044,
    K: int = 50,
    num_proc: int = 16,
    data_field: str = "text",
    max_samples: Optional[int] = None,
    overwrite_cache: bool = False,
) -> Dict:
    """Full trim pipeline: tokenize → count → classify → cascade → rebuild.

    Args:
        model_path: Path to model/tokenizer directory.
        data_path: Path to training data (JSON/JSONL/TXT).
        output_dir: Output directory for trimmed tokenizer + stats.
        freeze_idx: Token IDs below this are frozen.
        K: Frequency threshold for removal.
        num_proc: Number of tokenization workers.
        data_field: JSON field containing text.
        max_samples: Max documents to process (None = all).
        overwrite_cache: Force re-tokenization.

    Returns:
        Dict with trim results: removed_ids, removed_strs, old_to_new, stats.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load tokenizer ----
    print(f"Loading tokenizer from {model_path}...")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    vocab_size = len(tok.vocab)
    eos_id = tok.eos_token_id
    print(f"  Vocab size: {vocab_size}, EOS id: {eos_id}, freeze_idx: {freeze_idx}")

    # ---- Identify special IDs ----
    special_ids = get_special_token_ids(tok)
    print(f"  Special + filler IDs: {len(special_ids)}")

    trainable_ids = set(range(freeze_idx, vocab_size)) - special_ids
    print(f"  Trainable IDs: {len(trainable_ids)}")

    # ---- Load and tokenize dataset ----
    print(f"Loading dataset from {data_path}...")
    t0 = time.time()
    from datasets import load_dataset
    ext = os.path.splitext(data_path)[1].lower()
    if ext in [".jsonl", ".json"]:
        ds = load_dataset("json", data_files={"train": data_path}, split="train")
    elif ext in [".txt", ".text"]:
        ds = load_dataset("text", data_files={"train": data_path}, split="train")
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    print(f"  Loaded {len(ds)} docs in {time.time() - t0:.1f}s")

    print(f"Tokenizing with {num_proc} procs...")
    t0 = time.time()
    tokenize_fn = _get_tokenize_fn(tok)
    tokenized = ds.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=list(ds.features),
        load_from_cache_file=not overwrite_cache,
        desc="Tokenizing",
    )
    print(f"  Tokenized in {time.time() - t0:.1f}s")

    # ---- Count frequencies ----
    print("Counting token frequencies...")
    t0 = time.time()
    import pyarrow.compute as pc

    col = tokenized.data["input_ids"]
    arr = col.combine_chunks()
    flat = pc.list_flatten(arr)
    vc = pc.value_counts(flat)
    values = vc.field("values").to_pylist()
    counts = vc.field("counts").to_pylist()
    freq: Counter = Counter(dict(zip(values, counts)))
    total_tokens = sum(freq.values())
    print(f"  Counted in {time.time() - t0:.1f}s, {total_tokens:,} total tokens")

    # ---- Build merge tree ----
    print("Building merge tree...")
    merge_tree, vocab_str_to_id, safe_tokens = _load_merge_tree(tok)
    id_to_str = {v: k for k, v in vocab_str_to_id.items()}
    print(f"  Merge tree entries: {len(merge_tree)}")

    # ---- Classify tokens ----
    print("Classifying tokens...")
    token_class, child_to_parents = classify_tokens(
        vocab_str_to_id, merge_tree, trainable_ids, freeze_idx, eos_id,
    )
    class_counts = Counter(token_class.values())
    print(f"  Leaf: {class_counts.get('leaf', 0)}")
    print(f"  Terminal: {class_counts.get('terminal', 0)}")
    print(f"  Intermediate: {class_counts.get('intermediate', 0)}")

    # ---- Pre-removal stats ----
    trainable_freq = {tid: freq.get(tid, 0) for tid in trainable_ids}
    rare_before = sum(1 for v in trainable_freq.values() if 0 < v < K)
    zero_before = sum(1 for v in trainable_freq.values() if v == 0)
    print(f"\n  Before trimming:")
    print(f"    Trainable tokens: {len(trainable_ids)}")
    print(f"    Rare (0 < freq < {K}): {rare_before}")
    print(f"    Zero freq: {zero_before}")

    # ---- Cascade removal ----
    print(f"\nStarting cascade removal (K={K})...")
    t0 = time.time()
    removed_ids, removed_strs, iterations = cascade_remove(
        token_freq=trainable_freq,
        token_class=token_class,
        child_to_parents=child_to_parents,
        merge_tree=merge_tree,
        vocab_str_to_id=vocab_str_to_id,
        id_to_str=id_to_str,
        safe_tokens=safe_tokens,
        freeze_idx=freeze_idx,
        special_ids=special_ids,
        K=K,
    )
    print(f"  Cascade done in {time.time() - t0:.1f}s, {len(removed_ids)} tokens removed")

    # ---- Rebuild tokenizer ----
    print("\nBuilding trimmed tokenizer...")
    old_to_new, tok_data = rebuild_tokenizer(model_path, removed_strs, output_dir)

    new_vocab_size = len(tok_data["model"]["vocab"])

    # ---- Save outputs ----
    save_trim_outputs(
        output_dir=output_dir,
        tokenizer=tok,
        old_to_new=old_to_new,
        removed_ids=removed_ids,
        removed_strs=removed_strs,
        id_to_str=id_to_str,
        trainable_freq=trainable_freq,
        class_counts=class_counts,
        iterations=iterations,
        vocab_size=vocab_size,
        new_vocab_size=new_vocab_size,
        freeze_idx=freeze_idx,
        trainable_ids=trainable_ids,
        K=K,
        total_tokens=total_tokens,
        n_docs=len(tokenized),
        model_path=model_path,
        data_path=data_path,
        freq=freq,
    )

    # ---- Summary ----
    remaining_trainable = set(old_to_new[tid] for tid in trainable_ids if tid not in removed_ids)
    new_freq: Dict[int, int] = {}
    for old_tid, cnt in trainable_freq.items():
        if old_tid in old_to_new:
            new_freq[old_to_new[old_tid]] = cnt

    print_summary(
        vocab_size=vocab_size,
        new_vocab_size=new_vocab_size,
        removed_ids=removed_ids,
        iterations=iterations,
        trainable_ids=trainable_ids,
        remaining_trainable=remaining_trainable,
        K=K,
        trainable_freq=trainable_freq,
        new_freq=new_freq,
    )

    return {
        "removed_ids": removed_ids,
        "removed_strs": removed_strs,
        "old_to_new": old_to_new,
        "vocab_size": vocab_size,
        "new_vocab_size": new_vocab_size,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cascade-trim rare terminal tokens from tokenizer")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model/tokenizer")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data (JSON/JSONL)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for trimmed tokenizer + stats")
    parser.add_argument("--freeze_idx", type=int, default=248044, help="Token IDs below this are frozen")
    parser.add_argument("--K", type=int, default=50, help="Frequency threshold for removal")
    parser.add_argument("--num_proc", type=int, default=16, help="Number of tokenization workers")
    parser.add_argument("--data_field", type=str, default="text", help="JSON field containing text")
    parser.add_argument("--max_samples", type=int, default=None, help="Max documents to process")
    parser.add_argument("--overwrite_cache", action="store_true", help="Force re-tokenization (ignore cache)")
    args = parser.parse_args()

    trim_tokenizer(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        freeze_idx=args.freeze_idx,
        K=args.K,
        num_proc=args.num_proc,
        data_field=args.data_field,
        max_samples=args.max_samples,
        overwrite_cache=args.overwrite_cache,
    )


if __name__ == "__main__":
    main()
