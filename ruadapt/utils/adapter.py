"""Fix text-only LoRA adapter keys for the multimodal Qwen3.5 model.

Adapters trained with ``text_only=True`` are loaded via ``Qwen3_5ForCausalLM``:
the text backbone sits directly at ``model.layers.*``. In the full multimodal
``Qwen3_5ForConditionalGeneration`` the same backbone lives under
``model.language_model.*`` (alongside ``model.visual.*``). PEFT saves checkpoint
keys as ``base_model.model.<module path>``, so text-only adapters need the
``language_model.`` segment inserted after ``base_model.model.model.`` —
otherwise ``PeftModel.from_pretrained`` reports every key as missing
("Found missing adapter keys while loading the checkpoint") and merges such as
``scripts/merge_lora.py`` silently produce the unmerged base model.

Usage:
    python -m ruadapt.utils.adapter --input_dir /path/to/adapter \
        [--base_model Qwen/Qwen3.5-27B] [--output_dir /path/to/fixed] [--verify]

Default is in-place rewriting; ``--output_dir`` writes a copy instead.
``--verify`` instantiates the multimodal model on the meta device and checks
that the rewritten checkpoint keys exactly cover its PEFT target modules.
"""

import glob
import json
import os
import shutil
from typing import Dict, List, Optional, Set, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

TEXT_MODEL_PREFIX = "base_model.model.model."
LANGUAGE_SEGMENT = "language_model"
_UNTOUCHED_SEGMENTS = ("language_model", "visual")


def rewrite_adapter_key(key: str) -> str:
    """Insert the ``language_model.`` segment into a text-only PEFT key.

    Keys outside ``base_model.model.model.*`` (e.g. ``base_model.model.lm_head``,
    top-level in both classes), keys under ``model.visual.*`` and keys that are
    already multimodal are returned unchanged (idempotent).
    """
    if not key.startswith(TEXT_MODEL_PREFIX):
        return key
    segment = key[len(TEXT_MODEL_PREFIX):].split(".", 1)[0]
    if segment in _UNTOUCHED_SEGMENTS:
        return key
    return TEXT_MODEL_PREFIX + LANGUAGE_SEGMENT + "." + key[len(TEXT_MODEL_PREFIX):]


def _weight_files(adapter_dir: str) -> Tuple[List[str], List[str], Optional[str]]:
    """Return (safetensors shards, .bin shards, index file) of an adapter dir."""
    st_files = sorted(
        p for p in glob.glob(os.path.join(adapter_dir, "adapter_model*.safetensors"))
        if not p.endswith(".index.json")
    )
    bin_files = sorted(glob.glob(os.path.join(adapter_dir, "adapter_model*.bin")))
    bin_files = [p for p in bin_files if ".index" not in os.path.basename(p)]
    index_path = os.path.join(adapter_dir, "adapter_model.safetensors.index.json")
    index_file = index_path if os.path.exists(index_path) else None
    return st_files, bin_files, index_file


def checkpoint_keys(adapter_dir: str) -> Set[str]:
    """All parameter names stored in the adapter checkpoint (without loading tensors)."""
    st_files, bin_files, index_file = _weight_files(adapter_dir)
    if index_file is not None:
        with open(index_file, encoding="utf-8") as f:
            return set(json.load(f)["weight_map"].keys())
    keys: Set[str] = set()
    for path in st_files:
        with safe_open(path, framework="pt") as f:
            keys.update(f.keys())
    for path in bin_files:
        keys.update(torch.load(path, map_location="cpu", weights_only=True).keys())
    if not keys:
        raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")
    return keys


def _rewrite_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = rewrite_adapter_key(key)
        if new_key in out:
            raise ValueError(f"Key collision while rewriting: {key} -> {new_key}")
        out[new_key] = value
    return out


def _atomic_write_safetensors(state_dict: Dict[str, torch.Tensor], path: str, metadata: Optional[Dict[str, str]]):
    tmp = path + ".tmp"
    save_file(state_dict, tmp, metadata=metadata)
    os.replace(tmp, path)


def _fix_weight_files(st_files: List[str], bin_files: List[str], index_file: Optional[str]) -> int:
    """Rewrite keys in all checkpoint shards in place. Returns number of renamed keys."""
    renamed = 0
    for path in st_files:
        with safe_open(path, framework="pt") as f:
            metadata = f.metadata()
            state_dict = {k: f.get_tensor(k) for k in f.keys()}
        new_sd = _rewrite_state_dict(state_dict)
        renamed += sum(1 for k in state_dict if k != rewrite_adapter_key(k))
        _atomic_write_safetensors(new_sd, path, metadata)
    for path in bin_files:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        new_sd = _rewrite_state_dict(state_dict)
        renamed += sum(1 for k in state_dict if k != rewrite_adapter_key(k))
        tmp = path + ".tmp"
        torch.save(new_sd, tmp)
        os.replace(tmp, path)
    if index_file is not None:
        with open(index_file, encoding="utf-8") as f:
            index = json.load(f)
        index["weight_map"] = {rewrite_adapter_key(k): v for k, v in index["weight_map"].items()}
        tmp = index_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
        os.replace(tmp, index_file)
    return renamed


def _fix_adapter_config(adapter_dir: str, base_model: Optional[str]) -> Dict:
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    if base_model:
        config["base_model_name_or_path"] = base_model
    auto_mapping = config.get("auto_mapping")
    if isinstance(auto_mapping, dict) and auto_mapping.get("base_model_class") == "Qwen3_5ForCausalLM":
        auto_mapping["base_model_class"] = "Qwen3_5ForConditionalGeneration"
        config["auto_mapping"] = auto_mapping
    tmp = config_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    os.replace(tmp, config_path)
    return config


def expected_multimodal_keys(base_model_path: str, adapter_config: Dict) -> Set[str]:
    """Checkpoint keys PEFT expects when the adapter is applied to the multimodal model.

    Instantiates ``Qwen3_5ForConditionalGeneration`` on the meta device (no RAM/GPU)
    and resolves target modules with PEFT's own matching logic.
    """
    from dataclasses import fields as dataclass_fields

    from peft import LoraConfig
    from peft.tuners.tuners_utils import check_target_module_exists
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    hf_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    allowed = {f.name for f in dataclass_fields(LoraConfig)}
    lora_config = LoraConfig(**{k: v for k, v in adapter_config.items() if k in allowed})

    with torch.device("meta"):
        model = Qwen3_5ForConditionalGeneration(hf_config)

    modules_to_save = set(adapter_config.get("modules_to_save") or [])
    keys: Set[str] = set()
    for name, module in model.named_modules():
        if not name:
            continue
        short_name = name.rsplit(".", 1)[-1]
        if short_name in modules_to_save:
            keys.add(f"base_model.model.{name}.modules_to_save.default.weight")
            continue
        if check_target_module_exists(lora_config, name):
            if isinstance(module, torch.nn.Embedding):
                keys.add(f"base_model.model.{name}.lora_embedding_A.weight")
                keys.add(f"base_model.model.{name}.lora_embedding_B.weight")
            else:
                keys.add(f"base_model.model.{name}.lora_A.weight")
                keys.add(f"base_model.model.{name}.lora_B.weight")
    return keys


def verify_adapter(adapter_dir: str, base_model_path: str) -> Tuple[Set[str], Set[str]]:
    """Compare checkpoint keys against the multimodal model's expected keys.

    Returns (missing, unexpected); raises ValueError on any mismatch.
    """
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(config_path, encoding="utf-8") as f:
        adapter_config = json.load(f)
    actual = checkpoint_keys(adapter_dir)
    expected = expected_multimodal_keys(base_model_path, adapter_config)
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        raise ValueError(
            f"Adapter keys do not match the multimodal model ({base_model_path}).\n"
            f"Missing ({len(missing)}): {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}\n"
            f"Unexpected ({len(unexpected)}): {sorted(unexpected)[:10]}{'...' if len(unexpected) > 10 else ''}"
        )
    return missing, unexpected


def fix_adapter_dir(
    input_dir: str,
    base_model: Optional[str] = None,
    output_dir: Optional[str] = None,
    verify: bool = False,
) -> None:
    """Rewrite a text-only adapter checkpoint for the multimodal Qwen3.5 model.

    Args:
        input_dir: Adapter directory (adapter_config.json + adapter_model*.{safetensors,bin}).
        base_model: If given, written into ``base_model_name_or_path`` of the adapter config.
        output_dir: If given, write a fixed copy there instead of modifying input_dir in place.
        verify: After fixing, check keys against the multimodal model on the meta device.
    """
    if not os.path.isfile(os.path.join(input_dir, "adapter_config.json")):
        raise FileNotFoundError(f"No adapter_config.json in {input_dir}")

    target_dir = input_dir
    if output_dir is not None:
        target_dir = output_dir
        os.makedirs(target_dir, exist_ok=True)
        for entry in sorted(os.listdir(input_dir)):
            src = os.path.join(input_dir, entry)
            if os.path.isfile(src) and not entry.startswith("adapter_model"):
                shutil.copy2(src, os.path.join(target_dir, entry))
        st_files, bin_files, index_file = _weight_files(input_dir)
        for path in st_files + bin_files + ([index_file] if index_file else []):
            shutil.copy2(path, os.path.join(target_dir, os.path.basename(path)))

    config = _fix_adapter_config(target_dir, base_model)
    st_files, bin_files, index_file = _weight_files(target_dir)
    renamed = _fix_weight_files(st_files, bin_files, index_file)

    total = len(checkpoint_keys(target_dir))
    print(f"[fix_adapter] {input_dir} -> {target_dir}")
    print(f"[fix_adapter] renamed {renamed}/{total} keys")
    print(f"[fix_adapter] base_model_name_or_path: {config['base_model_name_or_path']}")

    if verify:
        base_model_path = base_model or config["base_model_name_or_path"]
        verify_adapter(target_dir, base_model_path)
        print(f"[fix_adapter] verified against {base_model_path}: all {total} keys match")


if __name__ == "__main__":
    import fire

    fire.Fire(fix_adapter_dir)
