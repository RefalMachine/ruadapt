import json
import os

import tiktoken
#from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
from typing import Dict, Optional

def bytes_to_unicode() -> dict[int, str]:
    """
    Returns the GPT-2 byte-to-unicode mapping.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

byte_encoder = bytes_to_unicode()

def token_bytes_to_string(b):
  return ''.join([byte_encoder[ord(char)] for char in b.decode('latin-1')])

# Adapted from https://github.com/openai/tiktoken/issues/60#issuecomment-1499977960
def bpe(mergeable_ranks: dict[bytes, int], token: bytes, max_rank: Optional[int] = None) -> list[bytes]:
  parts = [bytes([b]) for b in token]
  while True:
    min_idx = None
    min_rank = None
    for i, pair in enumerate(zip(parts[:-1], parts[1:])):
      rank = mergeable_ranks.get(pair[0] + pair[1])
      if rank is not None and (min_rank is None or rank < min_rank):
        min_idx = i
        min_rank = rank
    if min_rank is None or (max_rank is not None and min_rank >= max_rank):
      break
    assert min_idx is not None
    parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
  return parts

def generate_vocab_and_merges(encoder):
  mergeable_ranks = encoder._mergeable_ranks

  merges = []
  vocab = {}
  for token, rank in mergeable_ranks.items():
    vocab[token_bytes_to_string(token)] = rank

    if len(token) == 1:
      continue
    merged = tuple(bpe(mergeable_ranks, token, max_rank=rank))
    if len(merged) != 2:
      #print("RANK: ", rank)
      #print("MERGED PARTS:", [m.decode('utf-8', errors='replace') for m in merged], "| RAW BYTES:", merged)
      #print("ORIGINAL TOKEN:", token.decode('utf-8', errors='replace'), "| RAW BYTES:", token)
      #print('---'*10)
      continue

    merges.append(' '.join(map(token_bytes_to_string, merged)))

  # Also add special tokens
  vocab.update(encoder._special_tokens)

  return vocab, merges