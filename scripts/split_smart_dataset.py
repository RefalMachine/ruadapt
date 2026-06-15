#!/usr/bin/env python3
"""Split a smart K-coverage dataset into train / validation sets.

Shuffles documents randomly, takes `--val_size` examples for validation,
saves the two JSON files next to the source.

Usage:
    python scripts/split_smart_dataset.py \
        --input /workdir/pipeline_test/unigram128/smart_k100.json \
        --val_size 1000 \
        --seed 42
"""
import argparse
import json
import random


def main():
    parser = argparse.ArgumentParser(description="Split smart dataset into train/val")
    parser.add_argument("--input", required=True, help="Path to the smart dataset JSON")
    parser.add_argument("--val_size", type=int, default=1000,
                        help="Number of examples for validation (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--train_output", default=None,
                        help="Train output path (default: <input>_train.json)")
    parser.add_argument("--val_output", default=None,
                        help="Val output path (default: <input>_val.json)")
    args = parser.parse_args()

    print(f"Loading dataset from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    print(f"Total documents: {total}")

    if total < args.val_size:
        print(f"WARNING: dataset has only {total} docs, using all for validation.")
        args.val_size = total

    random.seed(args.seed)
    random.shuffle(data)

    val_data = data[:args.val_size]
    train_data = data[args.val_size:]

    stem = args.input.rsplit(".", 1)[0]
    train_path = args.train_output or f"{stem}_train.json"
    val_path = args.val_output or f"{stem}_val.json"

    for path, dataset in [(train_path, train_data), (val_path, val_data)]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(dataset)} docs -> {path}")

    print(f"\nDone. Train: {len(train_data)}, Val: {len(val_data)}")


if __name__ == "__main__":
    main()
