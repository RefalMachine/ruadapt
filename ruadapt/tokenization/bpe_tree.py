"""BPE merge tree — shared between tokenization and training.datasets."""

import random
from typing import List, Dict, Optional, Tuple, Set


def build_merge_tree(vocab: Dict[str, int], merges) -> Dict[str, Tuple[str, str]]:
    """Reconstruct the binary tree of BPE merges.

    Supports two merge formats:
    - List of "left right" strings (classic BPE)
    - List of ["left", "right"] lists (HF tokenizers >= 0.20)
    """
    tree = {}
    for merge_rule in merges:
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
        merged = left + right
        tree[merged] = (left, right)
    return tree


def recursive_split(
    token_str: str,
    tree: Dict[str, Tuple[str, str]],
    p_split: float = 0.5,
    force_split: bool = False,
    safe_tokens: Set[str] = None,
) -> List[str]:
    """Probabilistically split a token into its subwords based on the BPE merge tree.

    If safe_tokens is provided, aborts splits that would produce tokens
    missing from safe_tokens (e.g., broken UTF-8 bytes).
    """
    if token_str not in tree:
        return [token_str]

    left, right = tree[token_str]

    if safe_tokens is not None:
        if left not in safe_tokens or right not in safe_tokens:
            return [token_str]

    if force_split or random.random() < p_split:
        return recursive_split(left, tree, p_split, force_split=False, safe_tokens=safe_tokens) + recursive_split(
            right, tree, p_split, force_split=False, safe_tokens=safe_tokens
        )
    else:
        return [token_str]


def recursive_split_with_labels(
    token_str: str,
    tree: Dict[str, Tuple[str, str]],
    p_split: float = 0.5,
    force_split: bool = False,
    safe_tokens: Set[str] = None,
) -> Tuple[List[str], List[str]]:
    """Split token and return (subwords, labels) with hierarchical parent labels.

    Label for each subword:
    - First subword of each split: label = parent token (the one that was split)
    - Other subwords: label = themselves (leaf or further split)

    Example: ABCD -> AB + CD -> AB + C + D
        subwords: [AB, C, D]
        labels:   [ABCD, CD, D]
    """
    words: List[str] = []
    labels: List[str] = []
    _recursive_split_impl(token_str, tree, p_split, force_split, safe_tokens, words, labels)
    return words, labels


def _recursive_split_impl(
    token_str: str,
    tree: Dict[str, Tuple[str, str]],
    p_split: float,
    force_split: bool,
    safe_tokens: Optional[Set[str]],
    words_out: List[str],
    labels_out: List[str],
) -> None:
    """In-place recursive split — appends to words_out/labels_out."""
    if token_str not in tree:
        words_out.append(token_str)
        labels_out.append(token_str)
        return

    left, right = tree[token_str]

    if safe_tokens is not None:
        if left not in safe_tokens or right not in safe_tokens:
            words_out.append(token_str)
            labels_out.append(token_str)
            return

    if force_split or random.random() < p_split:
        start = len(words_out)
        _recursive_split_impl(left, tree, p_split, False, safe_tokens, words_out, labels_out)
        _recursive_split_impl(right, tree, p_split, False, safe_tokens, words_out, labels_out)
        labels_out[start] = token_str
    else:
        words_out.append(token_str)
        labels_out.append(token_str)
