"""Reference ready Arrow files without copying the token corpus a second time.

Manifests are local runtime artifacts: they contain absolute paths. Training
fails on missing/changed inputs or cache files instead of rebuilding the corpus
inside a large GPU allocation.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from datasets import Dataset, concatenate_datasets

from ruadapt.training.datasets.files import resolve_data_files


def _file_state(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _signature(cfg):
    data = asdict(cfg.data)
    for key in ("prepared_dataset_dir", "cache_dir", "overwrite_cache", "preprocessing_num_workers"):
        data.pop(key, None)
    return {
        "data": data, "model": asdict(cfg.model),
        "packing": asdict(cfg.packing) if cfg.packing else None,
        "sequence": asdict(cfg.sequence) if cfg.sequence else None,
        "sft": asdict(cfg.sft) if cfg.sft else None,
        "unified_dataset": asdict(cfg.unified_dataset) if cfg.unified_dataset else None,
        "freeze": asdict(cfg.freeze), "seed": cfg.training.seed,
        "dataset_factory": cfg.dataset_factory,
    }


def _inputs(cfg):
    return [(_file_state(path) if "://" not in path else {"url": path})
            for spec in (cfg.data.train_file, cfg.data.val_file) if spec is not None
            for path in resolve_data_files(spec)]


def _code_digest():
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    # Includes preprocessing, config resolution and shared tokenizer utilities.
    for name in ("training", "tokenization"):
        for path in sorted((root / name).rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def save_prepared(directory, cfg, train, evaluation):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    manifest = dict(version=1, signature=_signature(cfg), inputs=_inputs(cfg), code=_code_digest(), splits={})
    for name, dataset in (("train", train), ("eval", evaluation)):
        if dataset is None:
            manifest["splits"][name] = None
            continue
        if not isinstance(dataset, Dataset):
            raise ValueError("Prepared datasets require the shared Arrow dataset path (explicit packing/sequence)")
        if dataset._indices is not None:
            dataset = dataset.flatten_indices(cache_file_name=str(root / f"{name}.arrow"))
        if not dataset.cache_files:
            # Usually only a tiny in-memory/empty dataset needs serialization.
            dataset.save_to_disk(str(root / name))
            from datasets import load_from_disk
            dataset = load_from_disk(str(root / name))
        manifest["splits"][name] = {
            "rows": len(dataset), "files": [_file_state(entry["filename"]) for entry in dataset.cache_files],
        }
    temporary = root / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(root / "manifest.json")


def load_prepared(directory, cfg):
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    if manifest.get("version") != 1 or manifest["signature"] != _signature(cfg):
        raise ValueError("Prepared dataset config differs; run data preparation again")
    if manifest["code"] != _code_digest() or manifest["inputs"] != _inputs(cfg):
        raise ValueError("Prepared dataset code/input files changed; run data preparation again")
    result = []
    for name in ("train", "eval"):
        split = manifest["splits"][name]
        if split is None:
            result.append(None)
            continue
        parts = []
        for entry in split["files"]:
            if _file_state(entry["path"]) != entry:
                raise ValueError(f"Prepared Arrow file changed: {entry['path']}")
            parts.append(Dataset.from_file(entry["path"]))
        dataset = concatenate_datasets(parts)
        if len(dataset) != split["rows"]:
            raise ValueError(f"Prepared {name} row count changed")
        result.append(dataset)
    return tuple(result)
