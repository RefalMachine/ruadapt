"""Reusable data collators.

- SimpleStackCollator: stacks pre-padded tensors (for InMemoryPaddedDataset)
- PackedCollatorWithMask: stacks input_ids and labels (for packed datasets)
- DynamicPadCollator: pads to max length in batch (for SFT with variable-length sequences)
- PackedSFTCollator: stacks fixed-length packed SFT chunks (strict document packing)
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


class DynamicPadCollator:
    """Pad to max length in batch (not global max_length).

    For SFT with variable-length sequences. More memory efficient than
    pre-padding all sequences to global max_length.

    Args:
        pad_token_id: Token ID used for padding input_ids and attention_mask.
        label_pad_token_id: Value used for padding labels (default -100, ignored by loss).
        pad_to_multiple_of: Pad length to nearest multiple of this value.
    """

    def __init__(
        self,
        pad_token_id: int,
        label_pad_token_id: int = -100,
        pad_to_multiple_of: int = 8,
    ):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Find max length in batch
        max_len = 0
        for f in features:
            ids = f["input_ids"]
            max_len = max(max_len, len(ids) if not isinstance(ids, torch.Tensor) else ids.shape[0])

        # Round up to nearest multiple
        if self.pad_to_multiple_of and self.pad_to_multiple_of > 1:
            remainder = max_len % self.pad_to_multiple_of
            if remainder != 0:
                max_len += self.pad_to_multiple_of - remainder

        batch_size = len(features)
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), self.label_pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        for i, f in enumerate(features):
            ids = f["input_ids"]
            labs = f["labels"]
            attn = f["attention_mask"]

            # Handle both list and tensor
            if isinstance(ids, torch.Tensor):
                seq_len = ids.shape[0]
                input_ids[i, :seq_len] = ids
                labels[i, :seq_len] = labs
                attention_mask[i, :seq_len] = attn
            else:
                seq_len = len(ids)
                input_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long)
                labels[i, :seq_len] = torch.tensor(labs, dtype=torch.long)
                attention_mask[i, :seq_len] = torch.tensor(attn, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class PackedSFTCollator:
    """Stack fixed-length packed SFT chunks (strict document packing).

    Chunks are pre-packed by the dataset factory: every chunk has exactly
    pack_chunk_size tokens with position_ids reset per document, int32 seq_idx
    (document id per token) and int32 cu_seq_lens_q (cumulative segment lengths
    including the pad-tail segment). No attention_mask: packing isolation relies
    on position_ids (flash varlen path) + seq_idx/cu_seqlens (GatedDeltaNet).

    IMPORTANT: the dataset must be consumed with TrainingArguments
    remove_unused_columns=false, otherwise HF Trainer silently drops the
    seq_idx/cu_seq_lens_q columns (they are not in model.forward signature).
    """

    REQUIRED_KEYS = ("input_ids", "labels", "position_ids", "seq_idx", "cu_seq_lens_q")

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if len(features) != 1:
            raise ValueError(
                "PackedSFTCollator requires per_device_train_batch_size=1: FLA "
                "chunk kernels accept cu_seqlens only in flattened batch_size=1 form."
            )
        feature = features[0]
        batch: Dict[str, torch.Tensor] = {}
        for key in self.REQUIRED_KEYS:
            dtype = torch.int32 if key in ("seq_idx", "cu_seq_lens_q") else torch.long
            v = feature[key]
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v, dtype=dtype)
            else:
                v = v.to(dtype)
            if key == "cu_seq_lens_q":
                v = v.reshape(-1)  # FLA expects 1D cu_seqlens
            else:
                v = v.reshape(1, -1)
            batch[key] = v
        return batch
