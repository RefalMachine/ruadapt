from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ruadapt.training.config.schema import ModelConfig, SFTConfig, FreezeConfig
from ruadapt.training.core.validation import validate_training_config, validate_packing_model
from ruadapt.training.train import main


def config(**training):
    defaults = dict(per_device_train_batch_size=1, remove_unused_columns=False,
                    embed_weight_decay=0, fsdp=[], deepspeed=None)
    defaults.update(training)
    return SimpleNamespace(
        model=ModelConfig(model_name_or_path="unused", attn_implementation="flash_attention_3"),
        sft=SFTConfig(packing=True), freeze=FreezeConfig(), training=SimpleNamespace(**defaults),
    )


@pytest.mark.parametrize("backend", ["unknown", "eager"])
def test_backend_rejected_before_model_or_distributed_init(backend, monkeypatch):
    cfg = config()
    cfg.model.attn_implementation = backend
    monkeypatch.setattr("ruadapt.training.train.parse_config_from_json", lambda _: cfg)
    load = Mock()
    init = Mock()
    monkeypatch.setattr("ruadapt.training.train.load_model_and_tokenizer", load)
    monkeypatch.setattr("ruadapt.training.train.init_distributed", init)
    with pytest.raises(ValueError, match="Isolated packing requires"):
        main("ignored")
    load.assert_not_called()
    init.assert_not_called()


@pytest.mark.parametrize("overrides", [dict(remove_unused_columns=True)])
def test_packing_settings_rejected(overrides):
    with pytest.raises(ValueError):
        validate_training_config(config(**overrides))


def test_actual_backend_and_architecture_checked():
    cfg = config()
    text = SimpleNamespace(model_type="qwen3_5_text", _attn_implementation="flash_attention_3")
    for actual in (text, SimpleNamespace(text_config=text)):
        validate_packing_model(cfg, SimpleNamespace(config=actual))
    text._attn_implementation = "sdpa"
    with pytest.raises(ValueError, match="Loaded model"):
        validate_packing_model(cfg, SimpleNamespace(config=text))
    text._attn_implementation = "flash_attention_3"
    text.model_type = "llama"
    with pytest.raises(ValueError, match="Qwen3.5"):
        validate_packing_model(cfg, SimpleNamespace(config=text))


def test_dense_sdpa_still_allowed():
    cfg = config()
    cfg.sft.packing = False
    cfg.model.attn_implementation = "sdpa"
    validate_training_config(cfg)
    validate_packing_model(cfg, SimpleNamespace(config=SimpleNamespace(model_type="llama")))


@pytest.mark.parametrize("overrides", [dict(embed_weight_decay=0.01), dict(fsdp=["full_shard"]), dict(deepspeed="ds.json")])
def test_partial_freeze_constraints(overrides):
    cfg = config(**overrides)
    cfg.freeze = FreezeConfig(strategy="embed_only", freeze_idx=8)
    with pytest.raises(ValueError, match="Partial embedding freeze"):
        validate_training_config(cfg)


def test_isolated_sdpa_and_multiple_packs_supported():
    cfg = config(per_device_train_batch_size=2)
    cfg.model.attn_implementation = "sdpa"
    validate_training_config(cfg)
    for kind in ("qwen3", "qwen3_5_text"):
        validate_packing_model(cfg, SimpleNamespace(config=SimpleNamespace(model_type=kind, _attn_implementation="sdpa")))
