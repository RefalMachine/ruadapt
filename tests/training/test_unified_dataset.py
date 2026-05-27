"""Tests for packing functions and PackedDataset."""

import pytest
import torch
from datasets import Dataset

from ruadapt.training.datasets.packing import (
    PackedDataset,
    _try_substitute,
    compute_newline_ids,
    make_pack_fn,
)
from ruadapt.training.datasets.utils import _get_tokenize_fn


def _make_packed(tokenizer, texts, max_length=64, fragment_ratio=0.0, p_split=0.3, natural_boundaries=False):
    """Helper: tokenize → pack → PackedDataset (simulates factory path)."""
    fn = _get_tokenize_fn(tokenizer)
    batch = fn({"text": texts})
    tokenized = Dataset.from_dict({"input_ids": batch["input_ids"]})

    newline_ids = compute_newline_ids(tokenizer) if natural_boundaries else set()
    pack_fn = make_pack_fn(
        tokenizer=tokenizer,
        max_length=max_length,
        natural_boundaries=natural_boundaries,
        fragment_ratio=fragment_ratio,
        p_split=p_split,
        seed=42,
        newline_ids=newline_ids,
    )
    packed = tokenized.map(pack_fn, batched=True, num_proc=1)
    return PackedDataset(
        pre_packed_dataset=packed,
        tokenizer=tokenizer,
        max_length=max_length,
        fragment_ratio=fragment_ratio,
        p_split=p_split,
        natural_boundaries=natural_boundaries,
        n_documents=len(tokenized),
    )


class TestMakePackFn:
    def test_basic(self, tokenizer, sample_texts):
        ds = _make_packed(tokenizer, sample_texts, max_length=64)
        assert len(ds) > 0
        sample = ds[0]
        assert "input_ids" in sample
        assert "labels" in sample
        assert sample["input_ids"].shape[0] == 64

    def test_natural_boundaries(self, tokenizer, sample_texts):
        ds = _make_packed(tokenizer, sample_texts, max_length=64, natural_boundaries=True)
        assert len(ds) > 0
        assert ds.stats["natural_boundaries"] is True

    def test_substitution(self, tokenizer, sample_texts):
        ds = _make_packed(tokenizer, sample_texts, max_length=64, fragment_ratio=0.5, p_split=0.5)
        assert len(ds) > 0
        assert ds.stats["fragment_ratio"] == 0.5

    def test_labels_equal_input_ids_without_substitution(self, tokenizer, sample_texts):
        ds = _make_packed(tokenizer, sample_texts, max_length=64, fragment_ratio=0.0)
        for i in range(min(3, len(ds))):
            assert torch.equal(ds[i]["input_ids"], ds[i]["labels"])

    def test_stats_keys(self, tokenizer, sample_texts):
        ds = _make_packed(tokenizer, sample_texts, max_length=64)
        stats = ds.stats
        assert "total_tokens" in stats
        assert "usable_tokens" in stats
        assert "n_documents" in stats
        assert "n_chunks" in stats
        assert "max_length" in stats
        assert "token_frequency" in stats

    def test_n_documents(self, tokenizer, sample_texts):
        ds = _make_packed(tokenizer, sample_texts, max_length=64)
        assert ds.stats["n_documents"] == len(sample_texts)


class TestPackedDataset:
    def test_empty_dataset(self, tokenizer):
        empty = Dataset.from_dict({"input_ids": [], "labels": []})
        ds = PackedDataset(
            pre_packed_dataset=empty,
            tokenizer=tokenizer,
            max_length=64,
        )
        assert len(ds) == 0
        assert ds.stats["n_chunks"] == 0


class TestTrySubstitute:
    def test_try_substitute_basic(self):
        # Test that _try_substitute returns (sub_ids, label_ids) with hierarchical labels
        # "ab" splits into ["a", "b"] with labels ["ab", "b"]
        tree = {"ab": ("a", "b")}
        vocab = {"a": 0, "b": 1, "ab": 2}
        result = _try_substitute(2, "ab", tree, vocab, {"a", "b", "ab"}, 1.0)
        assert result is not None
        sub_ids, label_ids = result
        assert sub_ids == [0, 1]
        # First subword gets parent as label, second gets itself
        assert label_ids == [2, 1]

    def test_try_substitute_not_in_tree(self):
        result = _try_substitute(0, "x", {}, {}, set(), 1.0)
        assert result is None

    def test_try_substitute_none_str(self):
        result = _try_substitute(0, None, {}, {}, set(), 1.0)
        assert result is None

    def test_try_substitute_single_subword(self):
        # With force_split=True (first split always happens), p_split=0.0
        # only affects subsequent splits. "ab" → ["a", "b"] still happens.
        tree = {"ab": ("a", "b")}
        vocab = {"a": 0, "b": 1, "ab": 2}
        result = _try_substitute(2, "ab", tree, vocab, {"a", "b", "ab"}, 0.0)
        # First split is guaranteed, so result is not None
        assert result is not None
        sub_ids, label_ids = result
        assert sub_ids == [0, 1]
        assert label_ids == [2, 1]


class TestComputeNewlineIds:
    def test_returns_set(self, tokenizer):
        ids = compute_newline_ids(tokenizer)
        assert isinstance(ids, set)
        assert len(ids) > 0
