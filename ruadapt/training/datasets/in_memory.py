"""InMemoryPaddedDataset — pre-padded tensors in RAM.

From SFT script: converts HF dataset (lists of int) into contiguous tensors
padded to max_length. __getitem__ is a single tensor slice, zero overhead.
"""

from typing import Dict

import torch
from torch.utils.data import Dataset


class InMemoryPaddedDataset(Dataset):
    """Pre-padded dataset stored as contiguous tensors in RAM.

    Converts a HF Dataset (with input_ids, labels, attention_mask as lists of int)
    into padded torch tensors. Optimized for speed: no per-sample padding at
    __getitem__ time.

    Args:
        hf_dataset: HF Dataset with columns: input_ids, labels, attention_mask
            (each as list[int]).
        max_length: Target sequence length (pad/truncate to this).
        pad_token_id: Padding token ID for input_ids and attention_mask.
        label_pad_token_id: Padding value for labels (default -100, ignored by loss).
    """

    def __init__(
        self,
        hf_dataset,
        max_length: int,
        pad_token_id: int,
        label_pad_token_id: int = -100,
    ):
        # Single-process Python loop. For >100K samples, consider using
        # datasets.map() with num_proc for parallelism before passing here,
        # or torch.from_numpy() on pre-padded arrays.
        all_input_ids = []
        all_labels = []
        all_attention_mask = []

        for ex in hf_dataset:
            ids = ex["input_ids"]
            labs = ex["labels"]
            attn = ex["attention_mask"]
            pad_len = max_length - len(ids)

            all_input_ids.append(ids + [pad_token_id] * pad_len)
            all_labels.append(labs + [label_pad_token_id] * pad_len)
            all_attention_mask.append(attn + [0] * pad_len)

        self.input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        self.labels = torch.tensor(all_labels, dtype=torch.long)
        self.attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "attention_mask": self.attention_mask[idx],
        }
