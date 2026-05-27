"""Tests for factory protocol and load_factory."""

import pytest

from ruadapt.training.datasets.factory import load_factory, DatasetFactory, CollatorFactory


class TestLoadFactory:
    def test_load_factory_by_dotpath(self):
        cls = load_factory("ruadapt.training.datasets.unified_factory.UnifiedDatasetFactory")
        assert cls.__name__ == "UnifiedDatasetFactory"

    def test_load_factory_collator(self):
        cls = load_factory("ruadapt.training.datasets.unified_factory.UnifiedCollatorFactory")
        assert cls.__name__ == "UnifiedCollatorFactory"

    def test_load_factory_invalid(self):
        with pytest.raises((ValueError, ModuleNotFoundError, AttributeError)):
            load_factory("nonexistent.module.Factory")

    def test_load_factory_sft(self):
        cls = load_factory("ruadapt.training.datasets.sft_factory.SFTDatasetFactory")
        assert cls.__name__ == "SFTDatasetFactory"


class TestUnifiedDatasetFactory:
    def test_creates_dataset(self, tokenizer, config_dict):
        from ruadapt.training.train import parse_config_from_json
        import json
        import os

        # Write config to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_dict, f)
            config_path = f.name

        try:
            cfg = parse_config_from_json(config_path)
            factory_cls = load_factory(cfg.dataset_factory)
            factory = factory_cls()
            ds = factory.create_train(tokenizer, cfg)
            assert len(ds) > 0
        finally:
            os.unlink(config_path)

    def test_cpt_mode(self, tokenizer, config_dict):
        """fragment_ratio=0, natural_boundaries=False → CPT mode."""
        config_dict["unified_dataset"]["fragment_ratio"] = 0.0
        config_dict["unified_dataset"]["natural_boundaries"] = False

        import json, tempfile, os
        from ruadapt.training.train import parse_config_from_json

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_dict, f)
            config_path = f.name

        try:
            cfg = parse_config_from_json(config_path)
            factory_cls = load_factory(cfg.dataset_factory)
            factory = factory_cls()
            ds = factory.create_train(tokenizer, cfg)
            assert ds.stats["fragment_ratio"] == 0.0
        finally:
            os.unlink(config_path)

    def test_sub_mode(self, tokenizer, config_dict):
        """fragment_ratio>0 → substitution mode."""
        config_dict["unified_dataset"]["fragment_ratio"] = 0.3

        import json, tempfile, os
        from ruadapt.training.train import parse_config_from_json

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_dict, f)
            config_path = f.name

        try:
            cfg = parse_config_from_json(config_path)
            factory_cls = load_factory(cfg.dataset_factory)
            factory = factory_cls()
            ds = factory.create_train(tokenizer, cfg)
            assert ds.stats["fragment_ratio"] == 0.3
        finally:
            os.unlink(config_path)

    def test_eval_returns_none(self, tokenizer, config_dict):
        """Returns None when val_file is None."""
        config_dict["data"]["val_file"] = None

        import json, tempfile, os
        from ruadapt.training.train import parse_config_from_json

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_dict, f)
            config_path = f.name

        try:
            cfg = parse_config_from_json(config_path)
            factory_cls = load_factory(cfg.dataset_factory)
            factory = factory_cls()
            result = factory.create_eval(tokenizer, cfg)
            assert result is None
        finally:
            os.unlink(config_path)

    def test_max_samples(self, tokenizer, config_dict):
        """max_train_samples limits dataset."""
        config_dict["data"]["max_train_samples"] = 2

        import json, tempfile, os
        from ruadapt.training.train import parse_config_from_json

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_dict, f)
            config_path = f.name

        try:
            cfg = parse_config_from_json(config_path)
            factory_cls = load_factory(cfg.dataset_factory)
            factory = factory_cls()
            ds = factory.create_train(tokenizer, cfg)
            # With max_train_samples=2, at most 2 documents processed
            assert ds.stats["n_documents"] <= 2
        finally:
            os.unlink(config_path)
