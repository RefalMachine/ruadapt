import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ruadapt.training.config.schema import DataConfig, SFTConfig
from ruadapt.training.datasets.sft_factory import SFTDatasetFactory, _resolve_token_id


@pytest.mark.parametrize("explicit", [False, True])
def test_factory_masks_only_assistant_content(tmp_path, chat_tokenizer, explicit):
    path = tmp_path / "chat.json"
    path.write_text(json.dumps([{"messages": [
        {"role": "user", "content": "hello"}, {"role": "assistant", "content": "answer"},
    ]}]))
    sft = SFTConfig(max_tokens_count=64)
    if explicit:
        sft.im_start_token_id = 0
        sft.im_end_token_id = 1
    cfg = SimpleNamespace(data=DataConfig(train_file=str(path)), sft=sft)
    ds = SFTDatasetFactory().create_train(chat_tokenizer, cfg)
    assert len(ds) == 1
    assert ds[0]["input_ids"] == [0, 4, 3, 5, 1, 3, 0, 2, 3, 6, 1, 3]
    assert ds[0]["labels"] == [-100] * 9 + [6, 1, 3]


def test_explicit_zero_has_priority():
    tokenizer = Mock()
    assert _resolve_token_id(tokenizer, 0, "missing") == 0
    tokenizer.convert_tokens_to_ids.assert_not_called()


def test_missing_special_fails_before_data_load(monkeypatch, chat_tokenizer):
    load = Mock()
    monkeypatch.setattr("ruadapt.training.datasets.sft_factory.load_raw_dataset", load)
    unk_id = chat_tokenizer.unk_token_id
    monkeypatch.setattr(chat_tokenizer, "convert_tokens_to_ids", lambda _: unk_id)
    with pytest.raises(ValueError, match="im_start"):
        SFTDatasetFactory().create_train(chat_tokenizer, SimpleNamespace(data=None, sft=SFTConfig()))
    load.assert_not_called()


@pytest.mark.parametrize("text", ["assistant", "\n"])
def test_multitoken_boundary_rejected(chat_tokenizer, monkeypatch, text):
    encode = chat_tokenizer.encode
    monkeypatch.setattr(chat_tokenizer, "encode", lambda value, **kw: [2, 3] if value == text else encode(value, **kw))
    with pytest.raises(ValueError, match="single-token"):
        SFTDatasetFactory().create_train(chat_tokenizer, SimpleNamespace(data=None, sft=SFTConfig()))
