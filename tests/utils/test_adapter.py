"""Tests for ruadapt.utils.adapter: text_only -> multimodal adapter key fixing."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from ruadapt.utils.adapter import (
    checkpoint_keys,
    expected_multimodal_keys,
    fix_adapter_dir,
    rewrite_adapter_key,
    verify_adapter,
)


def test_rewrite_layer_keys():
    key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    assert rewrite_adapter_key(key) == "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"
    key = "base_model.model.model.layers.63.mlp.down_proj.lora_B.weight"
    assert rewrite_adapter_key(key) == "base_model.model.model.language_model.layers.63.mlp.down_proj.lora_B.weight"


def test_rewrite_embed_and_norm():
    key = "base_model.model.model.embed_tokens.modules_to_save.default.weight"
    assert rewrite_adapter_key(key) == "base_model.model.model.language_model.embed_tokens.modules_to_save.default.weight"
    key = "base_model.model.model.norm.modules_to_save.default.weight"
    assert rewrite_adapter_key(key) == "base_model.model.model.language_model.norm.modules_to_save.default.weight"


def test_rewrite_untouched_keys():
    # lm_head is top-level in both Qwen3_5ForCausalLM and Qwen3_5ForConditionalGeneration
    assert rewrite_adapter_key("base_model.model.lm_head.modules_to_save.default.weight") == \
        "base_model.model.lm_head.modules_to_save.default.weight"
    # visual branch already multimodal
    assert rewrite_adapter_key("base_model.model.model.visual.blocks.0.attn.qkv.lora_A.weight") == \
        "base_model.model.model.visual.blocks.0.attn.qkv.lora_A.weight"
    # already multimodal (idempotency)
    key = "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight"
    assert rewrite_adapter_key(key) == key
    # foreign keys
    assert rewrite_adapter_key("base_model.model.rotary_emb.inv_freq") == "base_model.model.rotary_emb.inv_freq"


ADAPTER_CONFIG = {
    "base_model_name_or_path": "fake/base",
    "auto_mapping": {
        "base_model_class": "Qwen3_5ForCausalLM",
        "parent_library": "transformers.models.qwen3_5.modeling_qwen3_5",
    },
    "peft_type": "LORA",
    "task_type": None,
    "r": 4,
    "lora_alpha": 4,
    "lora_dropout": 0.0,
    "target_modules": ["q_proj", "down_proj"],
    "modules_to_save": None,
    "bias": "none",
    "inference_mode": True,
}

TEXT_ONLY_WEIGHTS = {
    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(4, 8),
    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(8, 4),
    "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight": torch.randn(4, 8),
    "base_model.model.lm_head.modules_to_save.default.weight": torch.randn(8, 8),
    "base_model.model.model.embed_tokens.modules_to_save.default.weight": torch.randn(8, 8),
    "base_model.model.model.visual.blocks.0.attn.qkv.modules_to_save.default.weight": torch.randn(8, 8),
}


@pytest.fixture
def adapter_dir(tmp_path):
    path = tmp_path / "adapter"
    path.mkdir()
    save_file({k: v.clone() for k, v in TEXT_ONLY_WEIGHTS.items()}, str(path / "adapter_model.safetensors"),
              metadata={"format": "pt"})
    (path / "adapter_config.json").write_text(json.dumps(ADAPTER_CONFIG), encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    return str(path)


def test_fix_in_place(adapter_dir):
    fix_adapter_dir(adapter_dir, base_model="Qwen/Qwen3.5-27B")
    keys = checkpoint_keys(adapter_dir)
    assert "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight" in keys
    assert "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight" in keys
    assert "base_model.model.model.language_model.embed_tokens.modules_to_save.default.weight" in keys
    assert "base_model.model.lm_head.modules_to_save.default.weight" in keys
    assert "base_model.model.model.visual.blocks.0.attn.qkv.modules_to_save.default.weight" in keys
    assert len(keys) == len(TEXT_ONLY_WEIGHTS)

    sd = load_file(f"{adapter_dir}/adapter_model.safetensors")
    new_key = "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"
    assert torch.equal(sd[new_key], TEXT_ONLY_WEIGHTS["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"])

    config = json.load(open(f"{adapter_dir}/adapter_config.json"))
    assert config["base_model_name_or_path"] == "Qwen/Qwen3.5-27B"
    assert config["auto_mapping"]["base_model_class"] == "Qwen3_5ForConditionalGeneration"


def test_fix_in_place_idempotent(adapter_dir):
    fix_adapter_dir(adapter_dir)
    before = checkpoint_keys(adapter_dir)
    fix_adapter_dir(adapter_dir)
    assert checkpoint_keys(adapter_dir) == before


def test_fix_output_dir(adapter_dir, tmp_path):
    output_dir = str(tmp_path / "adapter_fixed")
    fix_adapter_dir(adapter_dir, output_dir=output_dir)
    # original untouched
    assert "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight" in checkpoint_keys(adapter_dir)
    # copy fixed and self-contained
    keys = checkpoint_keys(output_dir)
    assert "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight" in keys
    assert json.load(open(f"{output_dir}/adapter_config.json"))["auto_mapping"]["base_model_class"] == \
        "Qwen3_5ForConditionalGeneration"
    assert (tmp_path / "adapter_fixed" / "tokenizer.json").exists()


def test_fix_requires_adapter_config(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        fix_adapter_dir(str(empty))


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory):
    """Tiny multimodal Qwen3.5 model dir (config only) on disk."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig

    config = Qwen3_5Config(
        architectures=["Qwen3_5ForConditionalGeneration"],
        text_config=Qwen3_5TextConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=64,
            full_attention_interval=2,  # layer 0: linear attention, layer 1: full attention
        ),
    )
    path = tmp_path_factory.mktemp("tiny_base")
    config.save_pretrained(str(path))
    return str(path)


def test_expected_multimodal_keys(tiny_base):
    adapter_config = dict(ADAPTER_CONFIG, target_modules=["q_proj", "o_proj", "gate_proj", "down_proj"],
                          modules_to_save=["embed_tokens"])
    keys = expected_multimodal_keys(tiny_base, adapter_config)
    # layer 0 is linear attention: no self_attn targets there
    assert "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight" not in keys
    # layer 1 is full attention
    assert "base_model.model.model.language_model.layers.1.self_attn.q_proj.lora_A.weight" in keys
    assert "base_model.model.model.language_model.layers.1.self_attn.o_proj.lora_B.weight" in keys
    assert "base_model.model.model.language_model.layers.1.mlp.gate_proj.lora_A.weight" in keys
    assert "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_B.weight" in keys
    assert "base_model.model.model.language_model.embed_tokens.modules_to_save.default.weight" in keys
    # no vision keys: vision modules are qkv/proj/linear_fc*, not in target_modules
    assert not any(".visual." in k for k in keys)


def test_verify_adapter_detects_text_only_keys(adapter_dir, tiny_base):
    with pytest.raises(ValueError, match="do not match"):
        verify_adapter(adapter_dir, tiny_base)


def test_verify_adapter_passes_after_fix(adapter_dir, tiny_base):
    # rebuild checkpoint to exactly match the tiny model's targets
    config = dict(ADAPTER_CONFIG, target_modules=["q_proj", "down_proj"])
    weights = {
        "base_model.model.model.layers.1.self_attn.q_proj.lora_A.weight": torch.randn(4, 8),
        "base_model.model.model.layers.1.self_attn.q_proj.lora_B.weight": torch.randn(8, 4),
        "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight": torch.randn(4, 8),
        "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight": torch.randn(8, 4),
        "base_model.model.model.layers.1.mlp.down_proj.lora_A.weight": torch.randn(4, 8),
        "base_model.model.model.layers.1.mlp.down_proj.lora_B.weight": torch.randn(8, 4),
    }
    save_file(weights, f"{adapter_dir}/adapter_model.safetensors", metadata={"format": "pt"})
    with open(f"{adapter_dir}/adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f)
    fix_adapter_dir(adapter_dir, base_model=tiny_base, verify=True)
    missing, unexpected = verify_adapter(adapter_dir, tiny_base)
    assert not missing and not unexpected
