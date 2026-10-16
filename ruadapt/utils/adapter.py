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

To restore a full VLM checkpoint after a text-only save::

    python -m ruadapt.utils.adapter restore-vlm-config \
        --reference_model_path /path/to/original-vlm \
        --adapted_model_path /path/to/adapted-text-model \
        --restore_visual_weights
"""

import glob
import json
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

TEXT_MODEL_PREFIX = "base_model.model.model."
LANGUAGE_SEGMENT = "language_model"
_UNTOUCHED_SEGMENTS = ("language_model", "visual")
VISUAL_TOKENS = {
    "image_token_id": "<|image_pad|>",
    "video_token_id": "<|video_pad|>",
    "vision_start_token_id": "<|vision_start|>",
    "vision_end_token_id": "<|vision_end|>",
}
DROP_FROM_TEXT_CONFIG = {
    "architectures", "auto_map", "id2label", "image_token_id", "label2id", "text_config",
    "video_token_id", "vision_config", "vision_end_token_id", "vision_start_token_id",
}


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


def text_only_adapter_keys(adapter_dir: str) -> Set[str]:
    """Return Qwen3.5 text-backbone keys that still lack ``language_model``."""
    return {key for key in checkpoint_keys(adapter_dir) if rewrite_adapter_key(key) != key}


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


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _backup_once(path: Path, suffix: str) -> None:
    backup = path.with_name(path.name + suffix)
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)


def _model_weight_map(model_dir: Path) -> Tuple[Dict[str, str], Dict[str, Any]]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = load_json(index_path)
        return dict(index["weight_map"]), dict(index.get("metadata", {}))

    weight_map = {}
    model_files = [
        path for path in sorted(model_dir.glob("model*.safetensors"))
        if path.name != "model-visual-restored.safetensors"
    ]
    for path in model_files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in weight_map:
                    raise ValueError(f"Duplicate tensor {key} in model checkpoint")
                weight_map[key] = path.name
    if not weight_map:
        raise FileNotFoundError(f"No model safetensors found in {model_dir}")
    return weight_map, {"total_size": sum(path.stat().st_size for path in model_files)}


def _load_tensors(model_dir: Path, prefix: str) -> Dict[str, torch.Tensor]:
    weight_map, _ = _model_weight_map(model_dir)
    by_file: Dict[str, List[str]] = {}
    for key, filename in weight_map.items():
        if key.startswith(prefix):
            by_file.setdefault(filename, []).append(key)

    tensors = {}
    for filename, keys in by_file.items():
        with safe_open(model_dir / filename, framework="pt", device="cpu") as f:
            for key in keys:
                tensors[key] = f.get_tensor(key)
    return tensors


def merge_visual_weights(reference_model_path: str, adapted_model_path: str) -> None:
    """Add reference ``model.visual.*`` tensors as a separate indexed shard."""
    reference_dir = Path(reference_model_path)
    adapted_dir = Path(adapted_model_path)
    if not reference_dir.is_dir():
        raise ValueError("--restore_visual_weights requires a local reference model directory")

    visual_tensors = _load_tensors(reference_dir, "model.visual.")
    if not visual_tensors:
        raise ValueError(f"No model.visual.* tensors found in reference model: {reference_dir}")

    weight_map, metadata = _model_weight_map(adapted_dir)
    visual_shard = adapted_dir / "model-visual-restored.safetensors"
    already_indexed = visual_shard.name in set(weight_map.values())
    _atomic_write_safetensors(visual_tensors, str(visual_shard), {"format": "pt"})
    for key in visual_tensors:
        weight_map[key] = visual_shard.name

    index_path = adapted_dir/"model.safetensors.index.json"
    _backup_once(index_path, ".text_only.bak")
    visual_size = sum(tensor.numel() * tensor.element_size() for tensor in visual_tensors.values())
    if not already_indexed:
        metadata["total_size"] = int(metadata.get("total_size", 0)) + visual_size
    write_json(index_path, {"metadata": metadata, "weight_map": weight_map})
    print(f"[restore_vlm] restored {len(visual_tensors)} visual tensors into {visual_shard}")


def load_reference_config(reference_model_path: str) -> Dict[str, Any]:
    path = Path(reference_model_path)
    if path.is_dir():
        return load_json(path/"config.json")
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(reference_model_path, trust_remote_code=True).to_dict()


def get_token_ids(model_path: str) -> Dict[str, Optional[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    vocab = tokenizer.get_vocab()
    token_ids = {
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    for config_key, token in VISUAL_TOKENS.items():
        token_ids[config_key] = vocab.get(token)
    missing = [token for key, token in VISUAL_TOKENS.items() if token_ids[key] is None]
    if missing:
        raise ValueError(f"Adapted tokenizer is missing visual tokens: {', '.join(missing)}")
    return token_ids


def build_vlm_config(reference_config: Dict[str, Any], adapted_config: Dict[str, Any], token_ids: Dict[str, Optional[int]]) -> Dict[str, Any]:
    if "vision_config" not in reference_config:
        raise ValueError("Reference config must be from a VLM model and contain vision_config")

    text_config = deepcopy(adapted_config.get("text_config", {}))
    if not text_config:
        text_config = {key: deepcopy(value) for key, value in adapted_config.items() if key not in DROP_FROM_TEXT_CONFIG}

    text_config.setdefault("model_type", reference_config.get("text_config", {}).get("model_type", "qwen3_5_text"))
    for key in ("eos_token_id", "bos_token_id", "pad_token_id"):
        if token_ids.get(key) is not None:
            text_config[key] = token_ids[key]

    restored = deepcopy(reference_config)
    restored["text_config"] = text_config
    restored.pop("id2label", None)
    restored.pop("label2id", None)
    for key in ("dtype", "torch_dtype", "tie_word_embeddings"):
        if key in adapted_config:
            restored[key] = adapted_config[key]
    if "vocab_size" in text_config:
        restored["vocab_size"] = text_config["vocab_size"]
    for key in VISUAL_TOKENS:
        restored[key] = token_ids[key]
    return restored


def update_generation_config(model_path: Path, text_config: Dict[str, Any]) -> None:
    path = model_path/"generation_config.json"
    if not path.exists():
        return

    generation_config = load_json(path)
    for key in ("eos_token_id", "bos_token_id", "pad_token_id"):
        if key in text_config:
            generation_config[key] = text_config[key]
    write_json(path, generation_config)


def save_processor(reference_model_path: str, adapted_model_path: Path) -> None:
    from transformers import AutoProcessor, AutoTokenizer

    adapted_tokenizer = AutoTokenizer.from_pretrained(adapted_model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(reference_model_path, trust_remote_code=True)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer = adapted_tokenizer
    else:
        print("Warning: reference processor has no tokenizer attribute", file=sys.stderr)
    processor.save_pretrained(adapted_model_path)
    adapted_tokenizer.save_pretrained(adapted_model_path)


def validate_tokenizer_matches_config(model_path: Path) -> None:
    from transformers import AutoTokenizer

    tokenizer_size = len(AutoTokenizer.from_pretrained(model_path, trust_remote_code=True))
    config = load_json(model_path/"config.json")
    expected = {}
    if config.get("vocab_size") is not None:
        expected["config.vocab_size"] = config["vocab_size"]
    if config.get("text_config", {}).get("vocab_size") is not None:
        expected["config.text_config.vocab_size"] = config["text_config"]["vocab_size"]
    mismatches = {key: value for key, value in expected.items() if value != tokenizer_size}
    if mismatches:
        raise ValueError(f"Tokenizer size does not match model config: len(tokenizer)={tokenizer_size}, expected={mismatches}")


def restore_vlm_model(reference_model_path: str, adapted_model_path: str, restore_visual_weights: bool = False, save_processor_files: bool = True) -> None:
    """Restore multimodal config and optionally visual weights after a text-only save."""
    adapted_path = Path(adapted_model_path)
    config_path = adapted_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json in {adapted_path}")

    reference_config = load_reference_config(reference_model_path)
    adapted_config = load_json(config_path)
    restored = build_vlm_config(reference_config, adapted_config, get_token_ids(adapted_model_path))
    _backup_once(config_path, ".text_only.bak")
    write_json(config_path, restored)
    update_generation_config(adapted_path, restored["text_config"])
    if restore_visual_weights:
        merge_visual_weights(reference_model_path, adapted_model_path)
    if save_processor_files:
        try:
            save_processor(reference_model_path, adapted_path)
        except Exception as exc:
            print(f"Warning: processor was not saved: {exc}", file=sys.stderr)
    validate_tokenizer_matches_config(adapted_path)
    print(f"[restore_vlm] restored multimodal checkpoint metadata in {adapted_path}")


def restore_vlm_config(reference_model_path: str, adapted_model_path: str, no_processor: bool = False, restore_visual_weights: bool = False) -> None:
    """CLI-compatible wrapper for the former standalone restore_vlm_config utility."""
    restore_vlm_model(reference_model_path=reference_model_path,
                      adapted_model_path=adapted_model_path,
                      restore_visual_weights=restore_visual_weights,
                      save_processor_files=not no_processor)


def main() -> None:
    import fire

    restore_commands = {"restore-vlm", "restore_vlm", "restore-vlm-config", "restore_vlm_config",
                        "resolve-vlm-config", "resolve_vlm_config"}

    if len(sys.argv) > 1 and sys.argv[1] in restore_commands:
        sys.argv.pop(1)
        fire.Fire(restore_vlm_config)
    else:
        fire.Fire(fix_adapter_dir)


if __name__ == "__main__":
    main()
