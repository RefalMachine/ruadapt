"""Tests for BPE merge tree."""

import pytest

from ruadapt.tokenization.bpe_tree import build_merge_tree, recursive_split


class TestBuildMergeTree:
    def test_build_merge_tree(self, small_vocab):
        vocab, merges = small_vocab
        tree = build_merge_tree(vocab, merges)
        assert "ab" in tree
        assert tree["ab"] == ("a", "b")
        assert "abc" in tree
        assert tree["abc"] == ("ab", "c")

    def test_build_merge_tree_empty_merges(self):
        tree = build_merge_tree({"a": 0}, [])
        assert len(tree) == 0

    def test_build_merge_tree_list_format(self):
        """Supports list-of-lists merge format."""
        vocab = {"a": 0, "b": 1, "ab": 2}
        merges = [["a", "b"]]
        tree = build_merge_tree(vocab, merges)
        assert "ab" in tree
        assert tree["ab"] == ("a", "b")

    def test_build_merge_tree_invalid_merge(self):
        """Invalid merges are skipped."""
        vocab = {"a": 0, "b": 1}
        merges = ["invalid merge rule with spaces"]
        tree = build_merge_tree(vocab, merges)
        assert len(tree) == 0


class TestRecursiveSplit:
    def test_recursive_split_basic(self, small_vocab):
        vocab, merges = small_vocab
        tree = build_merge_tree(vocab, merges)
        result = recursive_split("abc", tree, p_split=1.0, force_split=True, safe_tokens=None)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_recursive_split_force(self, small_vocab):
        vocab, merges = small_vocab
        tree = build_merge_tree(vocab, merges)
        result = recursive_split("abc", tree, p_split=0.0, force_split=True, safe_tokens=None)
        assert len(result) > 1

    def test_recursive_split_no_force(self, small_vocab):
        vocab, merges = small_vocab
        tree = build_merge_tree(vocab, merges)
        result = recursive_split("abc", tree, p_split=0.0, force_split=False, safe_tokens=None)
        assert result == ["abc"]

    def test_recursive_split_probability(self, small_vocab):
        """p_split=0 -> no split, p_split=1 -> always split."""
        vocab, merges = small_vocab
        tree = build_merge_tree(vocab, merges)
        result = recursive_split("ab", tree, p_split=0.0, force_split=False, safe_tokens=None)
        assert result == ["ab"]
        result = recursive_split("ab", tree, p_split=1.0, force_split=False, safe_tokens=None)
        assert result == ["a", "b"]

    def test_recursive_split_safe_tokens(self, small_vocab):
        vocab, merges = small_vocab
        tree = build_merge_tree(vocab, merges)
        safe = {"abc"}
        result = recursive_split("abc", tree, p_split=1.0, force_split=True, safe_tokens=safe)
        assert result == ["abc"]

    def test_recursive_split_base_token(self):
        """Token not in tree returns as-is."""
        tree = {}
        result = recursive_split("x", tree, p_split=1.0, force_split=True, safe_tokens=None)
        assert result == ["x"]

    def test_recursive_split_leaf(self, small_vocab):
        """Leaf tokens (not in tree) return as-is."""
        vocab, merges = small_vocab
        tree = build_merge_tree(vocab, merges)
        result = recursive_split("c", tree, p_split=1.0, force_split=True, safe_tokens=None)
        assert result == ["c"]
