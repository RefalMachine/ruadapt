import json
import os

import tiktoken
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
from typing import Dict, Optional

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
    assert len(merged) == 2

    merges.append(' '.join(map(token_bytes_to_string, merged)))

  # Also add special tokens
  vocab.update(encoder._special_tokens)

  return vocab, merges