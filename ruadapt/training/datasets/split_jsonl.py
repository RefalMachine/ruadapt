"""Split local JSONL shards by stable document hash before tokenization/packing.

python -m ruadapt.training.datasets.split_jsonl --input '/data/part-*.jsonl' \
    --output-dir /data/legal_split --val-fraction 0.001 --seed 1337

Requires only the Python standard library. Keeps records and shard boundaries;
uses O(one document) RAM. The output directory must not exist.
"""

import argparse
import hashlib
import json
from pathlib import Path

from ruadapt.training.datasets.files import resolve_data_files


def validation_document(record, *, fraction, seed, group_field):
    value = record.get(group_field)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"Missing/empty split key: {group_field}")
    # Match CPT's stripping of leading/trailing whitespace for text keys.
    if isinstance(value, str):
        value = value.strip()
    key = json.dumps([seed, value], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    number = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    return number < int(fraction * (1 << 64))


def split_jsonl(inputs, output_dir, *, val_fraction=0.001, seed=1337, group_field="text"):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    files = resolve_data_files(inputs)
    if any("://" in path or Path(path).suffix != ".jsonl" for path in files):
        raise ValueError("Split preparation requires local .jsonl files")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    for split in ("train", "val"):
        (destination / split).mkdir()
    manifest = dict(algorithm="sha256-first64-json-seed-value-v1", seed=seed,
                    group_field=group_field, val_fraction=val_fraction,
                    train_documents=0, val_documents=0, shards=[])
    for index, filename in enumerate(files):
        counts = {"train": 0, "val": 0}
        name = f"part-{index:05d}.jsonl"
        # Byte-for-byte JSON records (apart from ensuring a newline), with all fields retained.
        with open(filename, encoding="utf-8") as source, \
                (destination / "train" / name).open("x", encoding="utf-8") as train, \
                (destination / "val" / name).open("x", encoding="utf-8") as val:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("Expected a JSON object")
                    is_val = validation_document(record, fraction=val_fraction, seed=seed, group_field=group_field)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{filename}:{line_number}: {exc}") from exc
                split = "val" if is_val else "train"
                (val if is_val else train).write(line if line.endswith("\n") else line + "\n")
                counts[split] += 1
        manifest["shards"].append(dict(source=filename, output=name, **counts))
        for split, count in counts.items():
            manifest[f"{split}_documents"] += count
        print(f"[{index + 1}/{len(files)}] {filename}: train={counts['train']}, val={counts['val']}", flush=True)
    if not manifest["train_documents"] or not manifest["val_documents"]:
        raise ValueError("Empty train or val split; choose a different fraction/seed and a NEW output directory")
    # Completion marker: do not use partially written output without this manifest.
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="Local JSONL paths/globs; quote glob patterns")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--group-field", default="text", help="Hash this field; use a stable case/document ID to group related records")
    args = parser.parse_args()
    result = split_jsonl(args.input, args.output_dir, val_fraction=args.val_fraction,
                         seed=args.seed, group_field=args.group_field)
    print(f"Completed: train={result['train_documents']}, val={result['val_documents']}; {args.output_dir}/manifest.json")


if __name__ == "__main__":
    main()
