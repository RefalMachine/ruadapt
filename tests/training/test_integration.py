"""Integration tests — end-to-end pipeline (no GPU required)."""

import pytest
import torch
from datasets import Dataset

from ruadapt.training.datasets.collators import PackedCollatorWithMask
from ruadapt.training.datasets.packing import PackedDataset, compute_newline_ids, make_pack_fn
from ruadapt.training.datasets.utils import _get_tokenize_fn


def _make_packed(tokenizer, texts, max_length=64, fragment_ratio=0.0, natural_boundaries=False):
    """Helper: tokenize → pack → PackedDataset."""
    fn = _get_tokenize_fn(tokenizer)
    batch = fn({"text": texts})
    tokenized = Dataset.from_dict({"input_ids": batch["input_ids"]})

    newline_ids = compute_newline_ids(tokenizer) if natural_boundaries else set()
    pack_fn = make_pack_fn(
        tokenizer=tokenizer,
        max_length=max_length,
        natural_boundaries=natural_boundaries,
        fragment_ratio=fragment_ratio,
        p_split=0.5,
        seed=42,
        newline_ids=newline_ids,
    )
    packed = tokenized.map(pack_fn, batched=True, num_proc=1)
    return PackedDataset(
        pre_packed_dataset=packed,
        tokenizer=tokenizer,
        max_length=max_length,
        fragment_ratio=fragment_ratio,
        natural_boundaries=natural_boundaries,
    )


class TestFullPipelineCPT:
    def test_full_pipeline_cpt(self, tokenizer, sample_texts):
        """tokenize → pack → collate → batch ready."""
        ds = _make_packed(tokenizer, sample_texts, max_length=64, fragment_ratio=0.0)
        assert len(ds) > 0

        collator = PackedCollatorWithMask()
        batch = collator([ds[0], ds[1]])
        assert "input_ids" in batch
        assert "labels" in batch
        assert batch["input_ids"].shape[0] == 2
        assert batch["input_ids"].shape[1] == 64

    def test_batch_has_no_nans(self, tokenizer, sample_texts):
        ds = _make_packed(tokenizer, sample_texts, max_length=64)
        collator = PackedCollatorWithMask()
        batch = collator([ds[0]])
        assert not torch.isnan(batch["input_ids"].float()).any()
        assert not torch.isnan(batch["labels"].float()).any()


class TestFullPipelineSub:
    def test_full_pipeline_sub(self, tokenizer, sample_texts):
        """Same with fragment_ratio>0."""
        ds = _make_packed(tokenizer, sample_texts, max_length=64, fragment_ratio=0.3)
        assert len(ds) > 0
        collator = PackedCollatorWithMask()
        batch = collator([ds[0]])
        assert "labels" in batch
        assert batch["labels"].shape == batch["input_ids"].shape


class TestFullPipelineCLM:
    def test_full_pipeline_clm(self, tokenizer, sample_texts):
        """Same with natural_boundaries=True."""
        ds = _make_packed(tokenizer, sample_texts, max_length=64, natural_boundaries=True)
        assert len(ds) > 0
        assert ds.stats["natural_boundaries"] is True


class TestParallelVsSequential:
    def test_parallel_tokenization(self, tokenizer, sample_texts, tmp_path):
        """num_proc=2 produces same token counts as num_proc=1."""
        tok_fn = _get_tokenize_fn(tokenizer)

        ds_seq = Dataset.from_dict({"text": sample_texts})
        ds_seq = ds_seq.map(tok_fn, batched=True, batch_size=1000, num_proc=1, remove_columns=["text"])

        ds_par = Dataset.from_dict({"text": sample_texts})
        ds_par = ds_par.map(tok_fn, batched=True, batch_size=1000, num_proc=2, remove_columns=["text"])

        total_seq = sum(len(r["input_ids"]) for r in ds_seq)
        total_par = sum(len(r["input_ids"]) for r in ds_par)
        assert total_seq == total_par
