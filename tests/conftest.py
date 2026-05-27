"""Shared fixtures for ruadapt tests."""

import json

import pytest


@pytest.fixture
def sample_texts():
    """Small text corpus for testing."""
    return [
        "Hello world. This is a test document with some content.",
        "Another document here. It has multiple sentences. For testing purposes.",
        "Short.",
        "A longer document that contains more text to test paragraph splitting "
        "and other tokenization features. " * 10,
    ]


@pytest.fixture
def sample_json_file(tmp_path, sample_texts):
    """Temporary JSON file with sample data."""
    path = tmp_path / "data.json"
    records = [{"text": t} for t in sample_texts]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f)
    return str(path)


@pytest.fixture
def sample_jsonl_file(tmp_path, sample_texts):
    """Temporary JSONL file with sample data."""
    path = tmp_path / "data.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for t in sample_texts:
            f.write(json.dumps({"text": t}) + "\n")
    return str(path)


@pytest.fixture
def small_vocab():
    """Minimal vocab + merges for BPE tree testing."""
    vocab = {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4}
    merges = ["a b", "ab c"]
    return vocab, merges
