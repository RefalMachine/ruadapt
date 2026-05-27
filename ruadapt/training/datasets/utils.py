"""Shared utilities for dataset modules.

Extracted from unified.py and factories to avoid duplication.
"""

import json
import os
from typing import Dict, List, Optional, Set, Tuple

import torch


def _log(msg: str, main_process_only: bool = True) -> None:
    """Print log message, optionally filtering only to main process (rank 0)."""
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        if main_process_only and rank != 0:
            return
        if rank != 0:
            msg = f"[rank {rank}] {msg}"
    print(msg)


def _is_rank_zero() -> bool:
    """Return True if this is rank 0 (or not in distributed mode)."""
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def _get_tokenizer_properties(tokenizer) -> Dict:
    """Detect if tokenizer prepends leading space to tokens."""
    leading_space = False
    space = None
    char = "1"
    tokens = tokenizer(char, add_special_tokens=False)["input_ids"]
    if len(tokens) > 1:
        space = tokenizer.convert_ids_to_tokens([tokens[0]])[0]
        leading_space = True
    else:
        token_str = tokenizer.convert_ids_to_tokens(tokens)[0]
        if len(token_str) != 1:
            space = token_str[0]
            leading_space = True

    space_token = tokenizer("1 ", add_special_tokens=False)["input_ids"]
    if leading_space:
        space_token = space_token[2]
    else:
        space_token = space_token[1]

    return {"force_leading_space": leading_space, "space": space}


def _custom_split(text: str, min_len: int = 10000) -> List[str]:
    """Split text at paragraph boundaries (\\n), keeping min_len chars per segment."""
    splitted = []
    prev_pos = 0
    pos = text.find("\n", prev_pos + min_len)
    while pos >= 0:
        s = text[pos]
        while s == "\n":
            pos += 1
            s = text[pos]
        splitted.append(text[prev_pos:pos])
        prev_pos = pos
        pos = text.find("\n", prev_pos + min_len)
    splitted.append(text[prev_pos:])
    return splitted


def _custom_tokenize_single(
    text: str, tokenizer, tokenizer_properties
) -> List[int]:
    """Tokenize a single document with BOS/EOS, handling long texts.

    For texts >100k characters, splits into paragraphs and tokenizes each
    separately to avoid tokenizer memory issues.
    """
    bos_token = tokenizer.bos_token or ""
    eos_token = tokenizer.eos_token or ""
    text_with_special = bos_token + text.strip() + eos_token

    if len(text_with_special) <= 100000:
        return tokenizer(text_with_special, add_special_tokens=False)["input_ids"]

    paragraphs = _custom_split(text_with_special)

    # First paragraph: tokenize normally
    first_ids = tokenizer(paragraphs[0], add_special_tokens=False)["input_ids"]

    # Subsequent paragraphs: handle leading space
    rest_ids = []
    for par in paragraphs[1:]:
        if tokenizer_properties["force_leading_space"]:
            t = tokenizer("\n" + par, add_special_tokens=False)["input_ids"]
            rest_ids.extend(t[2:])  # Strip fake leading space + dummy
        else:
            rest_ids.extend(tokenizer(par, add_special_tokens=False)["input_ids"])

    return first_ids + rest_ids


def _load_merge_tree(tokenizer) -> Tuple[Dict, Dict, Set[str]]:
    """Load BPE merge tree from tokenizer.json."""
    from ruadapt.tokenization.bpe_tree import build_merge_tree

    tok_path = None

    # Priority 1: tokenizer.json in model directory
    name_or_path = str(getattr(tokenizer, "name_or_path", ""))
    candidate = os.path.join(name_or_path, "tokenizer.json")
    if os.path.isfile(candidate):
        tok_path = candidate

    # Priority 2: vocab_file (only if it's a tokenizer.json format)
    if tok_path is None:
        vocab_file = getattr(tokenizer, "vocab_file", None)
        if vocab_file and os.path.isfile(str(vocab_file)):
            vf = str(vocab_file)
            try:
                with open(vf, "r", encoding="utf-8") as f:
                    probe = json.load(f)
                if isinstance(probe.get("model"), dict) and "merges" in probe.get("model", {}):
                    tok_path = vf
            except (json.JSONDecodeError, OSError):
                pass

    if tok_path is None:
        raise FileNotFoundError(
            "Cannot find tokenizer.json for merge tree construction."
        )

    with open(tok_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)

    model_data = tok_data.get("model", {})
    vocab_raw = model_data.get("vocab", {})
    merges = model_data.get("merges", [])

    vocab_str_to_id: Dict[str, int] = dict(vocab_raw)
    merge_tree = build_merge_tree(vocab_str_to_id, merges)

    safe_tokens: Set[str] = set()
    for t_str in vocab_str_to_id:
        decoded = tokenizer.convert_tokens_to_string([t_str])
        if "\ufffd" not in decoded:
            safe_tokens.add(t_str)

    return merge_tree, vocab_str_to_id, safe_tokens


def _token_to_str(tokenizer, token_id: int) -> Optional[str]:
    """Convert a token ID to its string representation."""
    try:
        token = tokenizer.convert_ids_to_tokens(token_id)
        if isinstance(token, list):
            return token[0] if token else None
        return token
    except Exception:
        return None


def _get_tokenize_fn(tokenizer, max_text_length: int = None):
    """Create tokenize function for dataset.map(batched=True).

    Returns a function compatible with HF dataset.map(batched=True, batch_size=1000).
    Tokenizes text with BOS/EOS, handles long texts via paragraph splitting.

    Uses batched tokenization for speed: one Rust call per batch instead of
    one per document.
    """
    tok_props = _get_tokenizer_properties(tokenizer)
    bos_token = tokenizer.bos_token or ""
    eos_token = tokenizer.eos_token or ""

    def tokenize_fn(examples):
        texts = examples["text"]
        n = len(texts)

        result_ids = [None] * n
        batch_indices = []
        batch_texts = []

        for i, text in enumerate(texts):
            if max_text_length and len(text) > max_text_length:
                result_ids[i] = []
            elif len(text) > 100000:
                result_ids[i] = _custom_tokenize_single(text, tokenizer, tok_props)
            else:
                batch_indices.append(i)
                batch_texts.append(bos_token + text.strip() + eos_token)

        if batch_texts:
            batch_encoded = tokenizer(
                batch_texts, add_special_tokens=False,
                padding=False, truncation=False,
            )
            for j, idx in enumerate(batch_indices):
                result_ids[idx] = batch_encoded["input_ids"][j]

        return {"input_ids": result_ids}

    return tokenize_fn


def _tokenize_texts_to_dataset(texts, tokenizer, max_text_length=None, domain_filter=None):
    """Tokenize raw texts into HF Dataset (single-process, no cache).

    For small datasets / debugging. Production uses factory with .map().

    Args:
        texts: List of text documents.
        tokenizer: HF tokenizer.
        max_text_length: Filter out texts longer than this (in characters).
        domain_filter: Optional callback fn(text) -> bool.

    Returns:
        HF Dataset with 'input_ids' column, or None if empty.
    """
    if domain_filter is not None:
        texts = [t for t in texts if domain_filter(t)]
    if max_text_length is not None:
        texts = [t for t in texts if len(t) <= max_text_length]
    if not texts:
        return None

    tokenize_fn = _get_tokenize_fn(tokenizer)
    batch = tokenize_fn({"text": texts})

    from datasets import Dataset
    return Dataset.from_dict({"input_ids": batch["input_ids"]})


def ensure_tokenized(raw, tokenize_fn, num_proc, overwrite_cache, is_main_process):
    """Tokenize HF dataset with caching. Only main rank writes cache.

    Flow:
    1. Non-main ranks hit barrier FIRST (wait for cache to be written)
    2. Main rank tokenizes, writes to HF datasets cache
    3. Main rank hits barrier (signals cache is ready)
    4. All ranks return tokenized dataset (from cache on non-main ranks)

    Args:
        raw: HF Dataset to tokenize.
        tokenize_fn: Tokenization function for .map().
        num_proc: Number of workers for .map().
        overwrite_cache: If True, re-tokenize even if cache exists.
            Only effective on main rank.
        is_main_process: Whether this rank should write cache.

    Returns:
        Tokenized HF Dataset.
    """
    import torch.distributed as dist

    column_names = list(raw.features)
    distributed = dist.is_initialized()

    if not is_main_process:
        if distributed:
            _log("Waiting at barrier for cache...", main_process_only=False)
            dist.barrier()
            _log("Cache ready, loading from cache...", main_process_only=False)
        return raw.map(
            tokenize_fn,
            batched=True,
            batch_size=1000,
            num_proc=num_proc,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Tokenizing (from cache)",
        )

    # Main rank: tokenize (overwrite_cache only on main)
    _log("[rank 0] Tokenizing dataset...")
    result = raw.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=column_names,
        load_from_cache_file=not overwrite_cache,
        desc="Tokenizing",
    )

    if distributed:
        _log("[rank 0] Tokenization done, signaling barrier...")
        dist.barrier()

    return result
