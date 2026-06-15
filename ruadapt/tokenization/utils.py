"""Token conversion, tokenizer property detection, and utility functions.

Shared concepts for the new-tokens pipeline:
  - freeze_idx: token ID boundary — IDs below are frozen base vocabulary
  - special_ids: all tokens that should be excluded from training targets
    (formally special + added tokens like free_token fillers)
  - new_token_ids: IDs >= freeze_idx that are NOT special (trainable new tokens)
  - filler_ids: added tokens that pad the vocab to 256-alignment (free_tokenN)
"""

from typing import Dict, List, Set

import torch
from torch import nn
from tqdm import tqdm


def convert_ascii_hex(token: str) -> int:
    """Convert a hex-encoded token like <0xXX> to its byte value."""
    return int(token[-2], 16) + 16 * int(token[-3], 16)


def if_hex(token: str) -> bool:
    """Check if token is a hex-encoded byte token."""
    return token.startswith("<0x") and token.endswith(">")


def simple_encode(token_str: str, tokenizer) -> List[int]:
    """Encode a token string without special tokens."""
    return tokenizer.encode(token_str, add_special_tokens=False)


def special_encode(token_str: str, tokenizer) -> List[int]:
    """Encode a token string, handling leading space via shift trick."""
    shift = len(tokenizer.encode("1", add_special_tokens=False))
    tokens = tokenizer.encode("1" + token_str, add_special_tokens=False)
    return tokens[shift:]


def get_mean_vec(token: str, tokenizer, embeddings, encode_func) -> torch.Tensor:
    """Get the mean embedding vector for a token across its subword decomposition."""
    tokens = encode_func(token, tokenizer)
    vector = embeddings[tokens].mean(axis=0)
    return vector


def convert_token_universal(token_str: str, tokenizer, vocab: dict = None, tokenizer_prop: dict = None) -> List[int]:
    """Convert a token string to its ID(s) using the tokenizer's internal model."""
    assert tokenizer.is_fast
    pre_tokenizer = tokenizer._tokenizer.pre_tokenizer
    if pre_tokenizer is not None:
        token_str = pre_tokenizer.pre_tokenize_str(token_str)
        token_str = "".join([t[0] for t in token_str])

    if tokenizer_prop and tokenizer_prop.get("space") == "▁":
        token_str = token_str.replace(" ", "▁")

    if if_hex(token_str) and vocab and token_str in vocab:
        return tokenizer.convert_tokens_to_ids([token_str])

    return [t.id for t in tokenizer._tokenizer.model.tokenize(token_str)]


def convert_token_to_string_universal(token: str, tokenizer_dst, tokeniser_src_vocab: dict = None, tokenizer_dst_properties: dict = None) -> str:
    """Convert a token to its string representation, handling hex tokens and leading spaces."""
    if if_hex(token):
        if tokeniser_src_vocab and token in tokeniser_src_vocab:
            return token

        token = chr(convert_ascii_hex(token))
        if tokeniser_src_vocab and token in tokeniser_src_vocab:
            return token

    token = [token]
    if tokenizer_dst_properties and tokenizer_dst_properties.get("force_leading_space"):
        token = [tokenizer_dst_properties["space"]] + token

    text_token = tokenizer_dst.convert_tokens_to_string(token)
    if len(text_token) == 1 and ord(text_token) == 65533:
        return token[-1]

    return text_token


def get_tokenizer_properties(tokenizer) -> Dict[str, object]:
    """Detect tokenizer properties: leading space behavior and space character."""
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
        assert len(space_token) == 3
        space_token = space_token[2]
        assert tokenizer.convert_ids_to_tokens([space_token])[0] == space
    else:
        assert len(space_token) == 2
        space_token = space_token[1]
        if space is None:
            space = tokenizer.convert_ids_to_tokens([space_token])[0]
        assert tokenizer.convert_ids_to_tokens([space_token])[0] == space

    return {"force_leading_space": leading_space, "space": space}


def get_first_diff_id(base_tokenizer_path: str, new_tokenizer_path: str) -> int:
    """Find the first token ID that differs between base and new tokenizers."""
    from transformers import AutoTokenizer

    print(f"Loading base tokenizer for comparison from {base_tokenizer_path}...")
    base_tok = AutoTokenizer.from_pretrained(base_tokenizer_path, trust_remote_code=True)
    new_tok = AutoTokenizer.from_pretrained(new_tokenizer_path, trust_remote_code=True)

    base_vocab = {v: k for k, v in base_tok.get_vocab().items()}
    new_vocab = {v: k for k, v in new_tok.get_vocab().items()}

    max_id = min(max(base_vocab.keys()), max(new_vocab.keys()))

    for i in range(max_id + 1):
        if base_vocab.get(i) != new_vocab.get(i):
            print(f"Vocabularies diverge at ID {i} (Base: {repr(base_vocab.get(i))}, New: {repr(new_vocab.get(i))})")
            return i

    return max_id + 1


def get_special_token_ids(tokenizer) -> Set[int]:
    """Extract all special token IDs including added tokens marked as special.

    Also includes free_tokens and other non-special added tokens,
    since they should be excluded from target vocabulary counts
    (consistent with trim_tokenizer.py logic that excludes IDs > eos_id).
    """
    special_ids = set()

    if hasattr(tokenizer, "all_special_ids") and tokenizer.all_special_ids:
        special_ids.update(tokenizer.all_special_ids)

    # All added tokens (special=True or not) should be excluded.
    # This covers free_tokens (<|free_token1|>, ...) which are added via
    # add_tokens() with special=False but are still non-trainable fillers.
    if hasattr(tokenizer, "added_tokens_decoder"):
        for t_id in tokenizer.added_tokens_decoder:
            special_ids.add(t_id)

    return special_ids


def get_filler_ids(tokenizer) -> Set[int]:
    """Extract filler token IDs (free_tokenN and similar padding tokens).

    These are added tokens whose content matches the ``<|free_tokenN|>`` pattern,
    used to pad vocabulary size to a multiple of 256.
    """
    filler_ids: Set[int] = set()
    if hasattr(tokenizer, "added_tokens_decoder"):
        for t_id, token_info in tokenizer.added_tokens_decoder.items():
            content = getattr(token_info, "content", "")
            if "free_token" in content:
                filler_ids.add(t_id)
    return filler_ids


def get_new_token_ids(tokenizer, freeze_idx: int) -> Set[int]:
    """Get IDs of new (non-frozen, non-special) tokens.

    These are tokens added during tokenizer extension that sit above
    ``freeze_idx`` and are not special/filler tokens.  They are the
    *trainable* vocabulary introduced by the extension.

    Args:
        tokenizer: HF tokenizer.
        freeze_idx: Token IDs below this are frozen base vocabulary.

    Returns:
        Set of new token IDs.
    """
    special_ids = get_special_token_ids(tokenizer)
    vocab_size = len(tokenizer.vocab) if hasattr(tokenizer, "vocab") else tokenizer.vocab_size
    return set(range(freeze_idx, vocab_size)) - special_ids


def get_trainable_ids(tokenizer, freeze_idx: int) -> Set[int]:
    """Get all trainable token IDs (new tokens + any non-special above freeze_idx).

    Equivalent to ``get_new_token_ids`` — kept as an alias for clarity in
    training code where the ``trainable`` naming is more natural.

    Args:
        tokenizer: HF tokenizer.
        freeze_idx: Token IDs below this are frozen base vocabulary.

    Returns:
        Set of trainable token IDs.
    """
    return get_new_token_ids(tokenizer, freeze_idx)
