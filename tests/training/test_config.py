"""Tests for config parsing."""

import json
import os

import pytest

from ruadapt.training.config.schema import (
    DataConfig,
    FreezeConfig,
    LoRAConfig,
    MainConfig,
    ModelConfig,
    TrainingConfig,
    UnifiedDatasetConfig,
)
from ruadapt.training.train import parse_config_from_json


class TestMainConfig:
    def test_parse_main_config(self, config_json_file):
        cfg = parse_config_from_json(config_json_file)
        assert isinstance(cfg, MainConfig)
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.data, DataConfig)
        assert isinstance(cfg.lora, LoRAConfig)
        assert isinstance(cfg.freeze, FreezeConfig)
        assert isinstance(cfg.training, TrainingConfig)
        assert isinstance(cfg.unified_dataset, UnifiedDatasetConfig)

    def test_data_config_defaults(self):
        cfg = DataConfig(train_file="dummy.json")
        assert cfg.cache_dir is None
        assert cfg.overwrite_cache is False
        assert cfg.streaming is False
        assert cfg.max_text_length is None

    def test_unknown_keys_rejected(self, tmp_path):
        path = tmp_path / "bad.json"
        with open(path, "w") as f:
            json.dump({"model": {}, "bad_key": True}, f)
        with pytest.raises(ValueError, match="Unknown top-level keys"):
            parse_config_from_json(str(path))

    def test_training_config_wsd(self, config_json_file):
        cfg = parse_config_from_json(config_json_file)
        assert cfg.training.wsd_constant_part == 0.85

    def test_unified_dataset_config(self, config_json_file):
        cfg = parse_config_from_json(config_json_file)
        ud = cfg.unified_dataset
        assert ud is not None
        assert ud.natural_boundaries is False
        assert ud.fragment_ratio == 0.0
        assert ud.p_split == 0.3

    def test_config_from_example_jsons(self):
        """All configs/*.json parse without error."""
        configs_dir = os.path.join(os.path.dirname(__file__), "..", "..", "configs")
        for name in os.listdir(configs_dir):
            if name.endswith(".json"):
                path = os.path.join(configs_dir, name)
                cfg = parse_config_from_json(path)
                assert isinstance(cfg, MainConfig), f"Failed for {name}"

    def test_smoke_configs_parse(self):
        """All configs/smoke/*.json parse without error."""
        smoke_dir = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "smoke")
        if not os.path.isdir(smoke_dir):
            pytest.skip("smoke dir not found")
        for name in os.listdir(smoke_dir):
            if name.endswith(".json"):
                path = os.path.join(smoke_dir, name)
                cfg = parse_config_from_json(path)
                assert isinstance(cfg, MainConfig), f"Failed for {name}"

    def test_lora_config_defaults(self):
        cfg = LoRAConfig()
        assert cfg.peft is False
        assert cfg.r == 8
        assert cfg.lora_alpha == 32.0
        assert cfg.lora_dropout == 0.0
        assert cfg.target_modules == ["q_proj", "v_proj"]
        assert cfg.modules_to_save is None

    def test_freeze_config_defaults(self):
        cfg = FreezeConfig()
        assert cfg.strategy == "none"
        assert cfg.freeze_idx is None
        assert cfg.unfreeze_modules is None

    def test_model_config_defaults(self):
        cfg = ModelConfig(model_name_or_path="test")
        assert cfg.torch_dtype == "bfloat16"
        assert cfg.trust_remote_code is False
        assert cfg.attn_implementation == "flash_attention_2"
