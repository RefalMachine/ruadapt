"""Tests for format converters in tokenization/convert.py."""

import pytest

from ruadapt.tokenization.convert import (
    bytes_to_unicode,
    token_bytes_to_string,
    bpe,
    check_ru_token,
    check_contains_digit,
    check_if_number,
)


class TestBytesToUnicode:
    def test_returns_dict(self):
        mapping = bytes_to_unicode()
        assert isinstance(mapping, dict)
        assert len(mapping) == 256

    def test_all_bytes_covered(self):
        mapping = bytes_to_unicode()
        for b in range(256):
            assert b in mapping

    def test_values_are_strings(self):
        mapping = bytes_to_unicode()
        for v in mapping.values():
            assert isinstance(v, str)


class TestTokenBytesToString:
    def test_ascii(self):
        result = token_bytes_to_string(b"hello")
        assert isinstance(result, str)
        assert len(result) == 5


class TestBPE:
    def test_single_byte(self):
        ranks = {b"a": 0, b"b": 1}
        result = bpe(ranks, b"a")
        assert result == [b"a"]

    def test_merge(self):
        ranks = {b"a": 0, b"b": 1, b"ab": 2}
        result = bpe(ranks, b"ab")
        assert result == [b"ab"]


class TestCheckRuToken:
    def test_russian(self):
        assert check_ru_token("привет") is True
        assert check_ru_token("Привет мир") is True

    def test_english(self):
        assert check_ru_token("hello") is False

    def test_mixed(self):
        assert check_ru_token("hello мир") is False

    def test_min_len(self):
        assert check_ru_token("а", min_len=1) is True
        assert check_ru_token("а", min_len=2) is False


class TestDigitChecks:
    def test_contains_digit(self):
        assert check_contains_digit("abc123") is True
        assert check_contains_digit("abc") is False

    def test_is_number(self):
        assert check_if_number("123") is True
        assert check_if_number("1") is False  # len > 1
        assert check_if_number("abc") is False
        assert check_if_number("12abc") is False
