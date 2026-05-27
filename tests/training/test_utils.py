"""Tests for datasets/utils.py."""

import pytest

from ruadapt.training.datasets.utils import (
    _custom_split,
    _get_tokenizer_properties,
    _is_rank_zero,
    _load_merge_tree,
    _token_to_str,
    _get_tokenize_fn,
    _tokenize_texts_to_dataset,
    ensure_tokenized,
)


class TestIsRankZero:
    def test_is_rank_zero_no_distributed(self):
        assert _is_rank_zero() is True


class TestLog:
    def test_log_no_distributed(self, capsys):
        from ruadapt.training.datasets.utils import _log

        _log("hello")
        captured = capsys.readouterr()
        assert "hello" in captured.out


class TestGetTokenizerProperties:
    def test_get_tokenizer_properties(self, tokenizer):
        props = _get_tokenizer_properties(tokenizer)
        assert "force_leading_space" in props
        assert "space" in props
        assert isinstance(props["force_leading_space"], bool)


class TestCustomSplit:
    def test_custom_split_basic(self):
        text = "a" * 5000 + "\n" + "b" * 5000 + "\n" + "c" * 5000
        parts = _custom_split(text, min_len=1000)
        assert len(parts) >= 2
        full = "".join(parts)
        assert len(full) == len(text)

    def test_custom_split_short_text(self):
        text = "short text without newlines"
        parts = _custom_split(text, min_len=100)
        assert len(parts) == 1
        assert parts[0] == text

    def test_custom_split_empty(self):
        parts = _custom_split("", min_len=100)
        assert len(parts) == 1


class TestLoadMergeTree:
    def test_load_merge_tree(self, tokenizer):
        merge_tree, vocab_str_to_id, safe_tokens = _load_merge_tree(tokenizer)
        assert isinstance(merge_tree, dict)
        assert isinstance(vocab_str_to_id, dict)
        assert isinstance(safe_tokens, set)
        assert len(merge_tree) > 0
        assert len(vocab_str_to_id) > 0
        assert len(safe_tokens) > 0


class TestTokenToStr:
    def test_token_to_str(self, tokenizer):
        result = _token_to_str(tokenizer, 0)
        assert result is None or isinstance(result, str)

    def test_token_to_str_invalid(self, tokenizer):
        result = _token_to_str(tokenizer, 999999999)
        assert result is None or isinstance(result, str)


class TestGetTokenizeFn:
    def test_tokenize_fn(self, tokenizer):
        fn = _get_tokenize_fn(tokenizer)
        result = fn({"text": ["Hello world"]})
        assert "input_ids" in result
        assert len(result["input_ids"]) == 1
        assert len(result["input_ids"][0]) > 0

    def test_tokenize_fn_batch(self, tokenizer):
        fn = _get_tokenize_fn(tokenizer)
        result = fn({"text": ["Hello world", "Another doc"]})
        assert len(result["input_ids"]) == 2

    def test_tokenize_fn_max_length(self, tokenizer):
        fn = _get_tokenize_fn(tokenizer, max_text_length=5)
        result = fn({"text": ["a" * 100]})
        assert result["input_ids"] == [[]]


class TestTokenizeTextsToDataset:
    def test_basic(self, tokenizer, sample_texts):
        ds = _tokenize_texts_to_dataset(sample_texts, tokenizer)
        assert ds is not None
        assert len(ds) > 0
        assert "input_ids" in ds.column_names

    def test_max_text_length(self, tokenizer, sample_texts):
        ds = _tokenize_texts_to_dataset(sample_texts, tokenizer, max_text_length=50)
        full_ds = _tokenize_texts_to_dataset(sample_texts, tokenizer)
        assert len(ds) <= len(full_ds)

    def test_domain_filter(self, tokenizer, sample_texts):
        ds = _tokenize_texts_to_dataset(
            sample_texts, tokenizer, domain_filter=lambda t: "Hello" in t
        )
        assert ds is not None
        full_ds = _tokenize_texts_to_dataset(sample_texts, tokenizer)
        assert len(ds) <= len(full_ds)

    def test_empty_returns_none(self, tokenizer):
        ds = _tokenize_texts_to_dataset([], tokenizer)
        assert ds is None

    def test_all_filtered_returns_none(self, tokenizer):
        ds = _tokenize_texts_to_dataset(
            ["no match"], tokenizer, domain_filter=lambda t: False
        )
        assert ds is None


class TestEnsureTokenized:
    def test_single_process(self, tokenizer, sample_texts):
        from datasets import Dataset

        raw = Dataset.from_dict({"text": sample_texts})
        tokenize_fn = _get_tokenize_fn(tokenizer)

        result = ensure_tokenized(
            raw, tokenize_fn, num_proc=1,
            overwrite_cache=False, is_main_process=True,
        )
        assert len(result) > 0
        assert "input_ids" in result.column_names

    def test_overwrite_cache(self, tokenizer, sample_texts):
        from datasets import Dataset

        raw = Dataset.from_dict({"text": sample_texts})
        tokenize_fn = _get_tokenize_fn(tokenizer)

        r1 = ensure_tokenized(
            raw, tokenize_fn, num_proc=1,
            overwrite_cache=True, is_main_process=True,
        )
        r2 = ensure_tokenized(
            raw, tokenize_fn, num_proc=1,
            overwrite_cache=True, is_main_process=True,
        )
        assert len(r1) == len(r2)
