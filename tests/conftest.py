"""Shared fixtures for ruadapt tests."""

import json

import pytest


@pytest.fixture(scope="session")
def local_tokenizer(tmp_path_factory):
    """Small byte-level BPE with real merges, built without Hub access."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    symbols = ["<pad>", "<eos>", "<unk>"] + sorted(pre_tokenizers.ByteLevel.alphabet())
    merges = [("h", "e"), ("l", "l"), ("he", "ll"), ("hell", "o")]
    symbols.extend(a + b for a, b in merges)
    backend = Tokenizer(models.BPE(vocab={s: i for i, s in enumerate(symbols)}, merges=merges, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    path = tmp_path_factory.mktemp("tokenizer")
    tok.save_pretrained(path)
    return PreTrainedTokenizerFast.from_pretrained(path)


@pytest.fixture
def chat_tokenizer():
    """Minimal ChatML tokenizer; newlines and assistant are single tokens."""
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {s: i for i, s in enumerate([
        "<|im_start|>", "<|im_end|>", "assistant", "\n", "user", "hello", "answer", "<unk>", "<pad>",
    ])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Split(Regex(r"\n"), behavior="isolated")
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>",
                                 eos_token="<|im_end|>", additional_special_tokens=["<|im_start|>"])
    tok.chat_template = "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' }}{% endfor %}"
    return tok


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
