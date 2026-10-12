"""Deterministic local data-file expansion shared by loaders and split preparation."""

import glob
from pathlib import Path


def resolve_data_files(spec):
    patterns = [spec] if isinstance(spec, str) else spec
    if not isinstance(patterns, (list, tuple)) or not patterns:
        raise ValueError("Data files must be a path/glob or nonempty list")
    files = []
    for pattern in patterns:
        if not isinstance(pattern, str):
            raise ValueError("Data file paths must be strings")
        # HF datasets handles remote URLs/globs itself.
        if "://" in pattern:
            files.append(pattern)
            continue
        matches = sorted(glob.glob(str(Path(pattern).expanduser())))
        if not matches or any(not Path(path).is_file() for path in matches):
            raise FileNotFoundError(f"No data files or non-file input: {pattern}")
        files.extend(str(Path(path).resolve()) for path in matches)
    if len(files) != len(set(files)):
        raise ValueError("Overlapping data patterns: a file would be loaded more than once")
    return files


def load_raw_dataset(data_config, split, *, json_only=False):
    from datasets import load_dataset

    spec = data_config.train_file if split == "train" else data_config.val_file
    if spec is None:
        return None
    files = resolve_data_files(spec)
    extensions = {Path(path).suffix.lower() for path in files}
    builders = {".json": "json", ".jsonl": "json", ".txt": "text"}
    formats = {builders.get(extension) for extension in extensions}
    if None in formats or len(formats) != 1 or (json_only and formats != {"json"}):
        raise ValueError(f"Unsupported/mixed data formats: {sorted(extensions)}")
    raw = load_dataset(formats.pop(), data_files={split: files}, split=split,
                       cache_dir=getattr(data_config, "cache_dir", None))
    limit = data_config.max_train_samples if split == "train" else data_config.max_val_samples
    if limit is not None:
        raw = raw.select(range(min(limit, len(raw))))
    return raw
