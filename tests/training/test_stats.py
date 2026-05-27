"""Tests for dataset stats module."""

import json
import os
from collections import Counter

import pytest

from ruadapt.training.datasets.stats import (
    print_dataset_stats,
    save_dataset_stats,
    plot_fragment_histogram,
    plot_token_frequency_histogram,
)


def _make_dummy_stats():
    """Create a minimal stats dict for testing."""
    return {
        "total_tokens": 1000,
        "usable_tokens": 900,
        "fragmented_tokens": 50,
        "target_substitutions": 50,
        "fragment_ratio_actual": 0.05,
        "fragment_length_distribution": {2: 30, 3: 15, 4: 5},
        "n_documents": 10,
        "n_chunks": 5,
        "max_length": 128,
        "natural_boundaries": False,
        "fragment_ratio": 0.1,
        "p_split": 0.3,
        "freeze_idx": None,
        "total_vocab_size": 100,
        "trainable_vocab_size": 80,
        "token_frequency": Counter({1: 100, 2: 50, 3: 30, 4: 20, 5: 10}),
    }


class DummyDataset:
    def __init__(self, stats):
        self.stats = stats


class TestPrintDatasetStats:
    def test_print_no_crash(self, tokenizer, capsys):
        stats = _make_dummy_stats()
        ds = DummyDataset(stats)
        print_dataset_stats(ds, tokenizer, top_k=5)
        captured = capsys.readouterr()
        assert "DATASET STATISTICS" in captured.out

    def test_print_empty_stats(self, tokenizer, capsys):
        ds = DummyDataset({})
        print_dataset_stats(ds, tokenizer)
        captured = capsys.readouterr()
        assert "No stats" in captured.out


class TestSaveDatasetStats:
    def test_save_creates_json(self, tmp_path):
        stats = _make_dummy_stats()
        path = str(tmp_path / "stats.json")
        save_dataset_stats(stats, path)
        assert os.path.exists(path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["total_tokens"] == 1000
        # Counter converted to dict
        assert isinstance(loaded["token_frequency"], dict)


class TestPlotFragmentHistogram:
    def test_creates_png(self, tmp_path):
        stats = _make_dummy_stats()
        path = str(tmp_path / "frag.png")
        result = plot_fragment_histogram(stats, save_path=path)
        if result is not None:  # matplotlib available
            assert os.path.exists(result)

    def test_empty_dist(self, tmp_path):
        stats = {"fragment_length_distribution": {}}
        result = plot_fragment_histogram(stats, save_path=str(tmp_path / "empty.png"))
        assert result is None


class TestPlotTokenFrequencyHistogram:
    def test_creates_png(self, tmp_path):
        stats = _make_dummy_stats()
        path = str(tmp_path / "freq.png")
        result = plot_token_frequency_histogram(stats, save_path=path)
        if result is not None:
            assert os.path.exists(result)

    def test_no_data(self, tmp_path):
        stats = {"token_frequency": Counter()}
        result = plot_token_frequency_histogram(stats, save_path=str(tmp_path / "none.png"))
        assert result is None
