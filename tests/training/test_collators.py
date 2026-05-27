"""Tests for data collators."""

import pytest
import torch

from ruadapt.training.datasets.collators import PackedCollatorWithMask, SimpleStackCollator


class TestPackedCollatorWithMask:
    def test_basic(self):
        """Basic collation with labels == input_ids."""
        collator = PackedCollatorWithMask()
        features = [
            {"input_ids": torch.tensor([1, 2, 3])},
            {"input_ids": torch.tensor([4, 5, 6])},
        ]
        result = collator(features)
        assert result["input_ids"].shape == (2, 3)
        assert result["labels"].shape == (2, 3)
        assert torch.equal(result["labels"], result["input_ids"])

    def test_preserves_labels(self):
        """Pre-computed labels preserved when present."""
        collator = PackedCollatorWithMask()
        features = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([10, 20, 30]),
            },
            {
                "input_ids": torch.tensor([4, 5, 6]),
                "labels": torch.tensor([40, 50, 60]),
            },
        ]
        result = collator(features)
        assert torch.equal(result["labels"][0], torch.tensor([10, 20, 30]))
        assert torch.equal(result["labels"][1], torch.tensor([40, 50, 60]))

    def test_no_labels_key(self):
        """Falls back to input_ids.clone() when no labels key."""
        collator = PackedCollatorWithMask()
        features = [
            {"input_ids": torch.tensor([1, 2, 3])},
        ]
        result = collator(features)
        assert torch.equal(result["labels"], result["input_ids"])


class TestSimpleStackCollator:
    def test_stack(self):
        collator = SimpleStackCollator()
        features = [
            {
                "input_ids": torch.tensor([1, 2]),
                "labels": torch.tensor([10, 20]),
                "attention_mask": torch.tensor([1, 1]),
            },
            {
                "input_ids": torch.tensor([3, 4]),
                "labels": torch.tensor([30, 40]),
                "attention_mask": torch.tensor([1, 0]),
            },
        ]
        result = collator(features)
        assert result["input_ids"].shape == (2, 2)
        assert result["labels"].shape == (2, 2)
        assert result["attention_mask"].shape == (2, 2)
        assert result["input_ids"][0][0].item() == 1
        assert result["input_ids"][1][1].item() == 4
