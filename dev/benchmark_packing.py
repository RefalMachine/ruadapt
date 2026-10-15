"""Compare the sequential reference and Arrow packing on a synthetic disk dataset.

Run: python -m dev.benchmark_packing --documents 10000 --workers 1 4
Uses a fresh temporary directory; measures packing plus Arrow writing, not tokenization.
"""
import argparse
import hashlib
from pathlib import Path
import tempfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from datasets import Dataset, Features, Sequence, Value, load_from_disk

from ruadapt.training.config.packing import PackingPolicy
from ruadapt.training.datasets.document_packing import _generate_packs, pack_dataset


def checksum(dataset):
    digests = [hashlib.sha256() for _ in range(6)]
    for batch in dataset.with_format("arrow").iter(batch_size=256):
        for i, name in enumerate(("input_ids", "labels", "segment_lengths")):
            digests[i * 2].update(pc.list_value_length(batch[name]).to_numpy().astype(np.int64).tobytes())
            digests[i * 2 + 1].update(pc.list_flatten(batch[name]).to_numpy().astype(np.int64).tobytes())
    return [digest.hexdigest() for digest in digests]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, default=10000)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4])
    args = parser.parse_args()
    lengths = np.random.default_rng(1337).integers(32, 8193, args.documents)
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
    tokens = int(offsets[-1])
    values = np.arange(tokens, dtype=np.int32) % 100000
    data = Dataset(pa.table({"input_ids": pa.ListArray.from_arrays(offsets, pa.array(values))}))
    policy = PackingPolicy("cpt", "isolated", 4096, "tokens", "dense", "pad", 4096, "split")
    with tempfile.TemporaryDirectory(prefix="ruadapt-packing-bench-") as temp:
        root = Path(temp)
        data.save_to_disk(str(root / "input"))
        data = load_from_disk(str(root / "input"))
        reference = None
        for workers in [0, *args.workers]:
            start = time.perf_counter()
            if workers == 0:
                result = Dataset.from_generator(
                    _generate_packs, gen_kwargs=dict(dataset=data, policy=policy, split_long=True, schema_version=2),
                    features=Features({name: Sequence(Value("int64")) for name in ("input_ids", "labels", "segment_lengths")}),
                    cache_dir=str(root / "reference"),
                )
            else:
                result = pack_dataset(data, policy, split_long=True, num_proc=workers,
                                      cache_dir=str(root / f"workers-{workers}"))
            elapsed = time.perf_counter() - start
            actual = checksum(result)
            if reference is None:
                reference = actual
            assert actual == reference, "Packed tokens/labels/segments differ from the reference"
            print(f"{'reference' if workers == 0 else f'Arrow workers={workers}'}: "
                  f"{tokens:,} tokens, {len(result):,} packs, {elapsed:.3f}s, {tokens / elapsed:,.0f} tok/s; exact match", flush=True)


if __name__ == "__main__":
    main()
