"""Tests for tokenization utility functions."""

import pytest

from ruadapt.tokenization.utils import (
    convert_ascii_hex,
    if_hex,
    get_tokenizer_properties,
    get_special_token_ids,
)


class TestHexConversion:
    def test_if_hex_positive(self):
        assert if_hex("<0x41>") is True
        assert if_hex("<0x0A>") is True

    def test_if_hex_negative(self):
        assert if_hex("hello") is False
        assert if_hex("<0x") is False
        assert if_hex("0x41>") is False

    def test_convert_ascii_hex(self):
        assert convert_ascii_hex("<0x41>") == 0x41
        assert convert_ascii_hex("<0x0A>") == 0x0A


class TestGetTokenizerProperties:
    def test_returns_dict(self):
        """Basic structure test (requires a real tokenizer)."""
        pytest.importorskip("transformers")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        props = get_tokenizer_properties(tokenizer)
        assert isinstance(props, dict)
        assert "force_leading_space" in props
        assert "space" in props


class TestGetSpecialTokenIds:
    def test_returns_set(self):
        pytest.importorskip("transformers")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        ids = get_special_token_ids(tokenizer)
        assert isinstance(ids, set)
        assert len(ids) > 0
