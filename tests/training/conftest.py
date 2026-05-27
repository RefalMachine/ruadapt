"""Shared fixtures for training tests."""

import json
import os

import pytest
import torch

MODEL_PATH = "/workdir/models/RuadaptQwen3.5-2B-Base-u128_cut64_min4_reldist_rdr0.3"


@pytest.fixture(scope="session")
def tokenizer():
    """Load local tokenizer for tests (session-scoped for speed)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


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
def config_dict(sample_json_file):
    """Minimal config dict for testing."""
    return {
        "model": {
            "model_name_or_path": MODEL_PATH,
            "torch_dtype": "bfloat16",
            "attn_implementation": "eager",
        },
        "data": {
            "train_file": sample_json_file,
            "block_size": 128,
            "preprocessing_num_workers": 2,
            "max_train_samples": 10,
        },
        "lora": {"peft": False},
        "freeze": {"strategy": "none"},
        "training": {
            "output_dir": "/tmp/test_output",
            "per_device_train_batch_size": 2,
            "max_steps": 1,
            "learning_rate": 1e-4,
            "bf16": False,
            "report_to": [],
        },
        "unified_dataset": {
            "natural_boundaries": False,
            "fragment_ratio": 0.0,
            "p_split": 0.3,
        },
        "dataset_factory": "ruadapt.training.datasets.unified_factory.UnifiedDatasetFactory",
        "collator_factory": "ruadapt.training.datasets.unified_factory.UnifiedCollatorFactory",
    }


@pytest.fixture
def config_json_file(tmp_path, config_dict):
    """Write config dict to JSON file."""
    path = tmp_path / "config.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f)
    return str(path)


@pytest.fixture
def small_vocab():
    """Minimal vocab + merges for BPE tree testing."""
    vocab = {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4}
    merges = ["a b", "ab c"]
    return vocab, merges
