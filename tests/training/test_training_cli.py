"""Exercise the installed console entry point and the module entry point."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from transformers import LlamaConfig, LlamaForCausalLM


@pytest.fixture(scope="module")
def installed_package(tmp_path_factory):
    root = Path(__file__).resolve().parents[2]
    temp = tmp_path_factory.mktemp("installed-cli")
    source = temp / "source"
    source.mkdir()
    shutil.copy2(root / "pyproject.toml", source / "pyproject.toml")
    shutil.copytree(root / "ruadapt", source / "ruadapt", ignore=shutil.ignore_patterns("__pycache__", "llmtf_open"))
    target = temp / "site"
    result = subprocess.run([
        sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "--no-index",
        "--target", str(target), str(source),
    ], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    env = dict(os.environ, PYTHONPATH=str(target), HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    return target, temp, env


@pytest.mark.parametrize("module", [False, True])
def test_help(installed_package, module):
    target, cwd, env = installed_package
    command = [sys.executable, "-m", "ruadapt.training.train"] if module else [str(target / "bin" / "ruadapt-train")]
    result = subprocess.run(command + ["--help"], cwd=cwd, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "config" in result.stdout + result.stderr


def test_installed_sft_prep_train_resume(installed_package, tmp_path, chat_tokenizer):
    target, cwd, env = installed_package
    model_dir = tmp_path / "base"
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(chat_tokenizer), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, pad_token_id=8, eos_token_id=1,
    ))
    model.save_pretrained(model_dir)
    chat_tokenizer.save_pretrained(model_dir)
    data = tmp_path / "sft.json"
    data.write_text(json.dumps([{"messages": [
        {"role": "user", "content": "hello"}, {"role": "assistant", "content": "answer"},
    ]}] * 8))
    output = tmp_path / "output"
    cfg = {
        "model": {"model_name_or_path": str(model_dir), "attn_implementation": "eager", "torch_dtype": "float32"},
        "data": {"train_file": str(data)},
        "sft": {"max_tokens_count": 64},
        "lora": {"peft": True, "r": 2, "target_modules": ["q_proj", "v_proj"]},
        "training": {"output_dir": str(output), "use_cpu": True, "max_steps": 2, "save_steps": 2,
                     "per_device_train_batch_size": 2, "report_to": [], "disable_tqdm": True,
                     "save_total_limit": 2},
        "dataset_factory": "ruadapt.training.datasets.sft_factory.SFTDatasetFactory",
        "collator_factory": "ruadapt.training.datasets.sft_factory.SFTCollatorFactory",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    command = [str(target / "bin" / "ruadapt-train"), "--config", str(path)]
    def run(extra=()):
        result = subprocess.run(command + list(extra), cwd=cwd, env=env, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    prepared = tmp_path / "prepared"
    run(["--data_prep_only", "--prepared_dataset_dir", str(prepared)])
    assert (prepared / "manifest.json").is_file()
    cfg["data"]["prepared_dataset_dir"] = str(prepared)
    path.write_text(json.dumps(cfg))
    assert not (output / "adapter_model.safetensors").exists()
    run()
    assert (output / "checkpoint-2" / "optimizer.pt").exists()
    assert (output / "final_lora_adapter" / "adapter_model.safetensors").exists()
    cfg["training"]["max_steps"] = 4
    path.write_text(json.dumps(cfg))
    result = run()
    assert "Resuming from checkpoint" in result.stdout
    state = json.loads((output / "checkpoint-4" / "trainer_state.json").read_text())
    assert state["global_step"] == 4
