"""Reusable data collators.

- SimpleStackCollator: stacks pre-padded tensors (for InMemoryPaddedDataset)
- PackedCollatorWithMask: stacks input_ids and labels (for packed datasets)
"""

from typing import Dict, List

import torch


class SimpleStackCollator:
    """Stack pre-padded tensors — no runtime padding needed.

    For use with InMemoryPaddedDataset where all sequences are already padded
    to the same length.
    """

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
        }


class PackedCollatorWithMask:
    """Stack input_ids and labels from packed datasets.

    If features contain pre-computed 'labels' (e.g. from target substitution),
    those are preserved. Otherwise labels = input_ids.
    """

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([f["input_ids"] for f in features])

        if "labels" in features[0]:
            labels = torch.stack([f["labels"] for f in features])
        else:
            labels = input_ids.clone()

        return {"input_ids": input_ids, "labels": labels}
