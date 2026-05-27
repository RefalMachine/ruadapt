"""Packed dataset and packing functions for CPT, CLM, and target substitution.

Pure functions for datasets.map():
    _try_substitute — BPE-Dropout substitution for a single token
    _try_targeted_substitute — targeted substitution for rare token exposure
    _pack_and_substitute_fn — batched map: concatenate → substitute → pack
    make_pack_fn — creates closure for datasets.map(pack_fn, batched=True)
    compute_newline_ids — collect newline token IDs from tokenizer

PackedDataset — thin torch Dataset wrapper over pre-packed HF Dataset.
    All heavy lifting (tokenization, packing, substitution) is done upstream
    via datasets.map() with caching. This class only converts columns to
    tensors and provides __getitem__ for the Trainer.
"""

import json
import os
import random
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
from torch.utils.data import Dataset
from ruadapt.tokenization.bpe_tree import recursive_split, recursive_split_with_labels
from ruadapt.training.datasets.utils import _load_merge_tree, _log


def _try_substitute(
    token_id: int,
    token_str: Optional[str],
    merge_tree: Dict,
    vocab_str_to_id: Dict[str, int],
    safe_tokens: Set[str],
    p_split: float,
) -> Optional[Tuple[List[int], List[int]]]:
    """Attempt BPE-Dropout substitution for a single token.

    Returns (sub_ids, label_ids) if substitution produces valid subwords,
    or None if no substitution is possible.

    label_ids[i] is:
    - parent token ID for the first subword of each split
    - subword ID itself for other subwords
    """
    if token_str is None or token_str not in merge_tree:
        return None

    subwords, labels = recursive_split_with_labels(
        token_str, merge_tree,
        p_split=p_split, force_split=True, safe_tokens=safe_tokens,
    )
    if len(subwords) <= 1:
        return None

    sub_ids = []
    label_ids = []
    for sw, lbl in zip(subwords, labels):
        sid = vocab_str_to_id.get(sw)
        lid = vocab_str_to_id.get(lbl)
        if sid is None or lid is None:
            return None
        sub_ids.append(sid)
        label_ids.append(lid)

    return sub_ids, label_ids


# ---------------------------------------------------------------------------
# Targeted substitution: expose rare tokens via ancestor splitting
# ---------------------------------------------------------------------------

def build_subtree_has_rare(
    merge_tree: Dict[str, Tuple[str, str]],
    rare_set: Set[str],
) -> Set[str]:
    """Compute set of tokens whose subtree contains at least one rare token.

    Uses memoized DFS. O(N) where N = len(merge_tree).
    """
    cache: Dict[str, bool] = {}

    def has_rare(token: str) -> bool:
        if token in cache:
            return cache[token]
        if token in rare_set:
            cache[token] = True
            return True
        if token not in merge_tree:
            cache[token] = False
            return False
        left, right = merge_tree[token]
        result = has_rare(left) or has_rare(right)
        cache[token] = result
        return result

    for token in merge_tree:
        has_rare(token)

    return set(k for k, v in cache.items() if v)


def load_targeted_substitution_data(
    trim_dir: str,
    merge_tree: Dict[str, Tuple[str, str]],
    vocab_str_to_id: Dict[str, int],
    K: int,
    min_parent_freq_ratio: float = 3.0,
) -> Tuple[Set[str], Set[str]]:
    """Load remaining rare tokens from trim output and build substitution sets.

    Args:
        trim_dir: path to trim_tokenizer output directory
        merge_tree: BPE merge tree
        vocab_str_to_id: vocab string -> id mapping
        K: rare token frequency threshold
        min_parent_freq_ratio: min parent freq as multiple of K

    Returns:
        parent_pool_set: set of token strings that should be substituted
        subtree_has_rare_set: set of tokens whose subtree contains rare tokens
    """
    stats_path = os.path.join(trim_dir, "trim_stats.json")
    with open(stats_path) as f:
        stats = json.load(f)

    remaining_rare = stats["remaining_rare"]
    rare_set: Set[str] = set()
    for t in remaining_rare:
        ts = t.get("token_str")
        if ts:
            rare_set.add(ts)

    _log(f"[targeted] Rare tokens: {len(rare_set)}")

    # Build subtree_has_rare set
    subtree_has_rare_set = build_subtree_has_rare(merge_tree, rare_set)
    _log(f"[targeted] Tokens with rare subtree: {len(subtree_has_rare_set)}")

    # Load frequencies to filter parents
    freq_path = os.path.join(trim_dir, "token_freq.json")
    freq_by_str: Dict[str, int] = {}
    if os.path.isfile(freq_path):
        with open(freq_path) as f:
            freq_data = json.load(f)
        id_to_str = {v: k for k, v in vocab_str_to_id.items()}
        freq_by_str = {id_to_str[int(k)]: v for k, v in freq_data["token_freq"].items() if int(k) in id_to_str}
    else:
        # Use remaining_rare frequencies only
        for t in remaining_rare:
            ts = t.get("token_str")
            if ts:
                freq_by_str[ts] = t["freq"]

    # Assume frozen tokens have high frequency
    freeze_idx = stats.get("freeze_idx", 248044)
    frozen_assumed_freq = stats.get("total_tokens", 1_500_000_000) // max(freeze_idx, 1)
    for tid in range(freeze_idx):
        ts = vocab_str_to_id.get(tid)  # Wrong direction, but we need id_to_str
    id_to_str = {v: k for k, v in vocab_str_to_id.items()}
    for tid in range(freeze_idx):
        ts = id_to_str.get(tid)
        if ts and ts not in freq_by_str:
            freq_by_str[ts] = frozen_assumed_freq

    # Parent pool: tokens with rare subtree AND freq >= min_parent_freq
    min_parent_freq = int(K * min_parent_freq_ratio)
    parent_pool_set = {
        t for t in subtree_has_rare_set
        if freq_by_str.get(t, 0) >= min_parent_freq
    }
    _log(f"[targeted] Parent pool (freq >= {min_parent_freq}): {len(parent_pool_set)}")

    return parent_pool_set, subtree_has_rare_set


def targeted_split(
    token_str: str,
    merge_tree: Dict[str, Tuple[str, str]],
    subtree_has_rare_set: Set[str],
    p_stop: float,
    safe_tokens: Set[str],
) -> List[str]:
    """Split token to expose rare children, stopping with probability p_stop at each node.

    Only splits nodes whose subtree contains rare tokens.
    """
    if token_str not in merge_tree:
        return [token_str]

    # Stop with probability p_stop — preserve current token
    if random.random() < p_stop:
        return [token_str]

    left, right = merge_tree[token_str]

    # Safe token check
    if safe_tokens is not None:
        if left not in safe_tokens or right not in safe_tokens:
            return [token_str]

    left_has_rare = left in subtree_has_rare_set
    right_has_rare = right in subtree_has_rare_set

    if not left_has_rare and not right_has_rare:
        return [token_str]

    result: List[str] = []
    if left_has_rare:
        result.extend(targeted_split(left, merge_tree, subtree_has_rare_set, p_stop, safe_tokens))
    else:
        result.append(left)

    if right_has_rare:
        result.extend(targeted_split(right, merge_tree, subtree_has_rare_set, p_stop, safe_tokens))
    else:
        result.append(right)

    return result


def _try_targeted_substitute(
    token_id: int,
    token_str: Optional[str],
    merge_tree: Dict[str, Tuple[str, str]],
    vocab_str_to_id: Dict[str, int],
    safe_tokens: Set[str],
    parent_pool_set: Set[str],
    subtree_has_rare_set: Set[str],
    fragment_ratio: float,
    p_split: float,
) -> Optional[Tuple[List[int], List[int]]]:
    """Attempt targeted substitution for a single token.

    Only applies to tokens in parent_pool_set. Splits to expose rare children.

    Returns (sub_ids, label_ids) if substitution produces valid subwords,
    or None if no substitution is applied.
    """
    if token_str is None or token_str not in parent_pool_set:
        return None

    if random.random() >= fragment_ratio:
        return None

    p_stop = 1.0 - p_split
    subwords = targeted_split(
        token_str, merge_tree, subtree_has_rare_set, p_stop, safe_tokens,
    )
    if len(subwords) <= 1:
        return None

    # Build labels: first subword gets parent, others get themselves
    labels = [token_str] + subwords[1:]

    sub_ids = []
    label_ids = []
    for sw, lbl in zip(subwords, labels):
        sid = vocab_str_to_id.get(sw)
        lid = vocab_str_to_id.get(lbl)
        if sid is None or lid is None:
            return None
        sub_ids.append(sid)
        label_ids.append(lid)

    return sub_ids, label_ids


def _pack_and_substitute_fn(
    examples: Dict[str, List],
    *,
    max_length: int,
    natural_boundaries: bool,
    fragment_ratio: float,
    p_split: float,
    seed: int,
    bos_id: Optional[int],
    eos_id: Optional[int],
    id_to_str: Dict[int, str],
    merge_tree: Dict,
    vocab_str_to_id: Dict[str, int],
    safe_tokens: Set[str],
    newline_ids: Set[int],
    substitution_method: str = "none",
    parent_pool_set: Optional[Set[str]] = None,
    subtree_has_rare_set: Optional[Set[str]] = None,
    substitutable_ids: Optional[Set[int]] = None,
    random_sub_ratio: float = 0.0,
) -> Dict[str, List[List[int]]]:
    """Batched map function: concatenate → substitute → pack into chunks.

    Called via datasets.map(pack_fn, batched=True). Deterministic via
    random.seed(seed) at each call — identical results on all ranks.

    substitution_method:
        "none" — no substitution
        "random" — BPE-Dropout (fragment_ratio selects tokens, p_split controls splitting)
        "targeted" — expose rare tokens (fragment_ratio selects parents, p_split controls depth)
        "hybrid" — targeted first, then random for remaining eligible tokens
            random_sub_ratio controls the probability multiplier for the random phase

    fragment_ratio: probability of applying substitution to a token
    p_split: probability of splitting at each merge tree node
        random: direct probability in recursive_split
        targeted: p_stop = 1 - p_split (probability of stopping)
    """
    random.seed(seed)
    from itertools import chain

    # Determine if any substitution is active
    do_random = substitution_method == "random" and fragment_ratio > 0
    do_targeted = substitution_method == "targeted" and parent_pool_set
    do_hybrid = substitution_method == "hybrid" and parent_pool_set

    # Speed up by using fast itertools.chain on C-level instead of sequential Python loops
    if not (do_random or do_targeted or do_hybrid):
        all_ids = list(chain(*examples["input_ids"]))
    else:
        all_ids = []
        for seq in examples["input_ids"]:
            all_ids.extend(seq)

    # For hybrid: precompute which tokens are in the parent pool (by ID)
    hybrid_pool_ids: Optional[Set[int]] = None
    if do_hybrid and parent_pool_set:
        hybrid_pool_ids = set()
        for ts in parent_pool_set:
            tid = vocab_str_to_id.get(ts)
            if tid is not None:
                hybrid_pool_ids.add(tid)

    if do_random or do_targeted or do_hybrid:
        input_ids: List[int] = []
        labels: List[int] = []
        n = len(all_ids)
        for i, tid in enumerate(all_ids):
            is_bos = (bos_id is not None and tid == bos_id and i == 0)
            is_eos = (eos_id is not None and tid == eos_id and i == n - 1)

            result = None
            if not is_bos and not is_eos and substitutable_ids and tid in substitutable_ids:
                ts = id_to_str.get(tid)
                if ts is not None:
                    if do_targeted:
                        result = _try_targeted_substitute(
                            tid, ts, merge_tree, vocab_str_to_id, safe_tokens,
                            parent_pool_set, subtree_has_rare_set,
                            fragment_ratio, p_split,
                        )
                    elif do_hybrid:
                        is_pool_token = hybrid_pool_ids and tid in hybrid_pool_ids
                        if is_pool_token:
                            # Phase 1: targeted substitution for pool tokens
                            result = _try_targeted_substitute(
                                tid, ts, merge_tree, vocab_str_to_id, safe_tokens,
                                parent_pool_set, subtree_has_rare_set,
                                fragment_ratio, p_split,
                            )
                        else:
                            # Phase 2: random substitution for non-pool tokens
                            # Probability = fragment_ratio * random_sub_ratio (e.g. 0.2 * 0.1 = 2%)
                            if random_sub_ratio > 0 and ts in merge_tree:
                                if random.random() < fragment_ratio * random_sub_ratio:
                                    result = _try_substitute(
                                        tid, ts, merge_tree, vocab_str_to_id, safe_tokens, p_split,
                                    )
                    elif do_random and random.random() < fragment_ratio:
                        result = _try_substitute(
                            tid, ts, merge_tree, vocab_str_to_id, safe_tokens, p_split,
                        )

            if result is not None:
                sub_ids, label_ids = result
                for si, li in zip(sub_ids, label_ids):
                    input_ids.append(si)
                    labels.append(li)
            else:
                input_ids.append(tid)
                labels.append(tid)
        has_substitutions = True
    else:
        input_ids = all_ids
        labels = all_ids
        has_substitutions = False

    # Pack into chunks
    if natural_boundaries and newline_ids:
        para_sep_ids = set()
        if eos_id is not None:
            para_sep_ids.add(eos_id)
        para_sep_ids.update(newline_ids)

        segment_start_tokens = set(newline_ids)
        if bos_id is not None:
            segment_start_tokens.add(bos_id)
        if eos_id is not None:
            segment_start_tokens.add(eos_id)

        total = len(input_ids)
        segments: List[Tuple[int, int]] = []
        start_idx = 0
        end_idx = start_idx + max_length

        while end_idx < total:
            while start_idx < total and input_ids[start_idx] not in segment_start_tokens:
                start_idx += 1
            if start_idx >= total:
                break
            while start_idx < total and input_ids[start_idx] in para_sep_ids:
                start_idx += 1
            if start_idx >= total:
                break
            end_idx = start_idx + max_length
            if end_idx <= total:
                segments.append((start_idx, end_idx))
            start_idx = end_idx

        if segments:
            out_ids = [input_ids[s:e] for s, e in segments]
            # Avoid redundant slicing if input_ids and labels are identical
            out_lbls = out_ids.copy() if not has_substitutions else [labels[s:e] for s, e in segments]
            return {"input_ids": out_ids, "labels": out_lbls}

    # Fallback: fixed-size packing
    usable = (len(input_ids) // max_length) * max_length
    out_ids = [input_ids[i:i + max_length] for i in range(0, usable, max_length)]
    # Avoid redundant slicing if input_ids and labels are identical
    out_lbls = out_ids.copy() if not has_substitutions else [labels[i:i + max_length] for i in range(0, usable, max_length)]
    return {"input_ids": out_ids, "labels": out_lbls}


def make_pack_fn(
    tokenizer: Any,
    max_length: int,
    natural_boundaries: bool,
    fragment_ratio: float,
    p_split: float,
    seed: int,
    newline_ids: Set[int],
    substitution_method: str = "none",
    min_parent_freq_ratio: float = 3.0,
    K: int = 50,
    trim_dir: Optional[str] = None,
    random_sub_ratio: float = 0.0,
) -> Callable:
    """Create a batched map function for packing + substitution.

    Returns a closure suitable for datasets.map(pack_fn, batched=True).
    All heavy objects (merge tree, vocab) are resolved once and captured
    in the closure.

    substitution_method:
        "none" — no substitution
        "random" — BPE-Dropout (fragment_ratio selects tokens, p_split controls splitting)
        "targeted" — expose rare tokens (fragment_ratio selects parents, p_split controls depth)
        "hybrid" — targeted first, then random for remaining eligible tokens

    fragment_ratio: probability of applying substitution to a token
    p_split: probability of splitting at each merge tree node
        random: direct probability in recursive_split
        targeted: p_stop = 1 - p_split (probability of stopping)
    random_sub_ratio: probability multiplier for random phase in hybrid mode
    """
    bos_id = getattr(tokenizer, "bos_token_id", None)
    eos_id = tokenizer.eos_token_id

    # Load merge tree if any substitution method is active
    need_merge_tree = (substitution_method == "random" and fragment_ratio > 0) or \
                      substitution_method in ("targeted", "hybrid")

    if need_merge_tree:
        merge_tree, vocab_str_to_id, safe_tokens = _load_merge_tree(tokenizer)
        id_to_str = {v: k for k, v in vocab_str_to_id.items()}
    else:
        merge_tree = {}
        vocab_str_to_id = {}
        safe_tokens = set()
        id_to_str = {}

    # Build targeted substitution sets if needed
    parent_pool_set: Optional[Set[str]] = None
    subtree_has_rare_set: Optional[Set[str]] = None

    if substitution_method in ("targeted", "hybrid"):
        if trim_dir is None:
            raise ValueError("targeted substitution requires trim_dir in config")
        parent_pool_set, subtree_has_rare_set = load_targeted_substitution_data(
            trim_dir, merge_tree, vocab_str_to_id, K, min_parent_freq_ratio,
        )

    # Pre-compute set of token IDs that can be substituted (skip frozen/special)
    substitutable_ids: Optional[Set[int]] = None
    if substitution_method == "random" and fragment_ratio > 0:
        substitutable_ids = set(vocab_str_to_id.get(ts) for ts in merge_tree)
        substitutable_ids.discard(None)
    elif substitution_method in ("targeted", "hybrid") and parent_pool_set:
        substitutable_ids = set(vocab_str_to_id.get(ts) for ts in parent_pool_set)
        substitutable_ids.discard(None)
        # For hybrid: also include all tokens in merge_tree for the random phase
        if substitution_method == "hybrid":
            random_ids = set(vocab_str_to_id.get(ts) for ts in merge_tree)
            random_ids.discard(None)
            substitutable_ids.update(random_ids)

    def pack_fn(examples, idx=None):
        # Incorporate batch/shard index into the seed for perfect determinism in parallel execution
        local_seed = seed
        if idx is not None:
            # If idx is a list of indices, use the first one, otherwise use it directly
            batch_offset = idx[0] if isinstance(idx, list) else idx
            local_seed = seed + batch_offset

        return _pack_and_substitute_fn(
            examples,
            max_length=max_length,
            natural_boundaries=natural_boundaries,
            fragment_ratio=fragment_ratio,
            p_split=p_split,
            seed=local_seed,
            bos_id=bos_id,
            eos_id=eos_id,
            id_to_str=id_to_str,
            merge_tree=merge_tree,
            vocab_str_to_id=vocab_str_to_id,
            safe_tokens=safe_tokens,
            newline_ids=newline_ids,
            substitution_method=substitution_method,
            parent_pool_set=parent_pool_set,
            subtree_has_rare_set=subtree_has_rare_set,
            substitutable_ids=substitutable_ids,
            random_sub_ratio=random_sub_ratio,
        )

    return pack_fn


def compute_newline_ids(tokenizer) -> Set[int]:
    """Collect token IDs that decode to strings containing newline."""
    newline_ids: Set[int] = set()
    vocab = tokenizer.vocab
    for token_str in vocab:
        decoded = tokenizer.convert_tokens_to_string([token_str])
        if "\n" in decoded:
            newline_ids.add(vocab[token_str])
    return newline_ids


class PackedDataset(Dataset):
    """Thin torch Dataset over pre-packed HF Dataset (input_ids + labels).

    All heavy lifting (tokenization, packing, substitution) is done upstream
    via datasets.map() with caching. This class only:
    1. Converts HF Dataset columns to contiguous torch.Tensor
    2. Provides __getitem__ for the Trainer
    3. Computes token_frequency stats for diagnostics

    Args:
        pre_packed_dataset: HF Dataset with 'input_ids' and 'labels' columns.
        tokenizer: HF tokenizer (for vocab size in stats).
        max_length: Chunk size (for stats).
        natural_boundaries: Whether natural boundary packing was used (for stats).
        fragment_ratio: Configured fragment ratio (for stats).
        p_split: Configured p_split (for stats).
        freeze_idx: If set, filter token_frequency to tokens >= freeze_idx.
        n_documents: Original document count before packing (for stats).
    """

    def __init__(
        self,
        pre_packed_dataset,
        tokenizer: Any = None,
        max_length: int = 512,
        natural_boundaries: bool = False,
        fragment_ratio: float = 0.0,
        p_split: float = 0.3,
        freeze_idx: Optional[int] = None,
        n_documents: Optional[int] = None,
        compute_token_freq: bool = False,
    ):
        self.max_length = max_length
        self.natural_boundaries = natural_boundaries
        self.fragment_ratio = fragment_ratio
        self.p_split = p_split
        self.freeze_idx = freeze_idx

        n_chunks = len(pre_packed_dataset)

        if n_chunks == 0:
            self.data = torch.empty(0, max_length, dtype=torch.long)
            self.labels = torch.empty(0, max_length, dtype=torch.long)
            self.total_tokens = 0
            _log("Packed: 0 chunks — empty dataset")
            self._init_stats_empty(tokenizer, n_documents)
            return

        import time
        t_start = time.time()

        # Extract underlying PyArrow arrays directly.
        # This completely bypasses HF datasets formatting engine (avoiding the torchvision bug)
        # and converts Arrow memory to NumPy in microseconds with zero copies!
        import numpy as np
        table = pre_packed_dataset.data
        
        # Bypassing datasets __getitem__ to avoid calling broken torchvision.io imports
        all_ids_np = np.stack(table.column("input_ids").to_numpy())
        all_labels_np = np.stack(table.column("labels").to_numpy())
        
        self.data = torch.from_numpy(all_ids_np).long()
        self.labels = torch.from_numpy(all_labels_np).long()

        t_convert = time.time() - t_start
        _log(f"Arrow to Tensor zero-copy convert took: {t_convert:.4f} seconds")

        usable = n_chunks * max_length
        self.total_tokens = usable
        _log(f"Packed: {n_chunks} chunks x {max_length} = {usable:,} tokens")

        if compute_token_freq:
            t_stats_start = time.time()
            token_freq: Counter = Counter()
            # Iterate over numpy array directly for speed
            for chunk in all_ids_np:
                token_freq.update(chunk)

            if freeze_idx is not None:
                trainable_token_freq = Counter(
                    {tid: cnt for tid, cnt in token_freq.items() if tid >= freeze_idx}
                )
            else:
                trainable_token_freq = token_freq

            trainable_vocab_size = len(trainable_token_freq)
            t_stats = time.time() - t_stats_start
            _log(f"Token frequency statistics calculation took: {t_stats:.4f} seconds")
        else:
            token_freq = Counter()
            trainable_token_freq = Counter()
            trainable_vocab_size = 0

        # Compute substitution stats from input_ids vs labels
        # fragmented: positions where input != label (first subword of each substitution)
        n_fragmented = int((self.data != self.labels).sum().item())
        fragment_ratio_actual = n_fragmented / max(usable, 1)

        self.stats = {
            "total_tokens": usable,
            "usable_tokens": usable,
            "fragmented_tokens": n_fragmented,
            "target_substitutions": n_fragmented,
            "fragment_ratio_actual": fragment_ratio_actual,
            "fragment_length_distribution": {},
            "n_documents": n_documents if n_documents is not None else 0,
            "n_chunks": n_chunks,
            "max_length": max_length,
            "natural_boundaries": natural_boundaries,
            "fragment_ratio": fragment_ratio,
            "p_split": p_split,
            "freeze_idx": freeze_idx,
            "total_vocab_size": len(tokenizer.vocab) if tokenizer else 0,
            "trainable_vocab_size": trainable_vocab_size,
            "token_frequency": trainable_token_freq,
        }

    def _init_stats_empty(self, tokenizer, n_documents):
        self.stats = {
            "total_tokens": 0,
            "usable_tokens": 0,
            "fragmented_tokens": 0,
            "target_substitutions": 0,
            "fragment_ratio_actual": 0.0,
            "fragment_length_distribution": {},
            "n_documents": n_documents if n_documents is not None else 0,
            "n_chunks": 0,
            "max_length": self.max_length,
            "natural_boundaries": self.natural_boundaries,
            "fragment_ratio": self.fragment_ratio,
            "p_split": self.p_split,
            "freeze_idx": self.freeze_idx,
            "total_vocab_size": len(tokenizer.vocab) if tokenizer else 0,
            "trainable_vocab_size": 0,
            "token_frequency": Counter(),
        }

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.data[i],
            "labels": self.labels[i],
        }
