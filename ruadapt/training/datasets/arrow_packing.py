"""Dense packing over Arrow buffers, parallelized by global output-token ranges.

Only document lengths are scanned globally. Workers copy numeric buffers (not
Python token lists), and own complete output packs. A document crossing a worker
boundary therefore has exactly the same segmentation as in iter_packs.
"""
from pathlib import Path
import json

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from datasets import Dataset, Features, Sequence, Value, concatenate_datasets, config as datasets_config
from datasets.fingerprint import Hasher
from filelock import FileLock

# Implementation cache version; does not change the training/checkpoint policy.
ARROW_PACKING_VERSION = 1
PACKS_PER_TASK = 256


def _lengths(dataset):
    columns = ["input_ids"] + (["labels"] if "labels" in dataset.column_names else [])
    lengths = []
    for batch in dataset.select_columns(columns).with_format("arrow").iter(batch_size=8192):
        x = pc.list_value_length(batch["input_ids"])
        if x.null_count:
            raise ValueError("input_ids must not contain null documents")
        if "labels" in columns:
            y = pc.list_value_length(batch["labels"])
            if y.null_count or not pc.all(pc.equal(x, y)).as_py():
                raise ValueError("input_ids and labels must have the same length")
        lengths.append(x.to_numpy().astype(np.int64, copy=False))
    return np.concatenate(lengths) if lengths else np.empty(0, dtype=np.int64)


def _pack_ranges(batch, *, source, offsets, policy, sequence_length, overflow):
    """Dataset.map callback returning Arrow directly, avoiding per-token Python objects."""
    start, stop = batch["start"][0], batch["stop"][0]
    first = int(np.searchsorted(offsets, start, side="right") - 1)
    last = int(np.searchsorted(offsets, stop, side="left"))
    size = stop - start
    ids = np.empty(size, dtype=np.int64)
    labels = np.empty(size, dtype=np.int64)
    boundaries = [0]
    row_index = first
    table = source.with_format("arrow")[first:last]
    for rows in table.to_batches():
        xs = rows.column("input_ids")
        ys = rows.column("labels") if "labels" in rows.schema.names else xs
        xvalues, yvalues = xs.values, ys.values
        xoffsets, yoffsets = xs.offsets.to_numpy(), ys.offsets.to_numpy()
        for i in range(rows.num_rows):
            doc_start, doc_end = int(offsets[row_index]), int(offsets[row_index + 1])
            row_index += 1
            left, right = max(start, doc_start), min(stop, doc_end)
            if right <= left:
                continue
            a, b = left - doc_start, right - doc_start
            dest = slice(left - start, right - start)
            ids[dest] = xvalues.slice(int(xoffsets[i]) + a, b - a).to_numpy()
            labels[dest] = yvalues.slice(int(yoffsets[i]) + a, b - a).to_numpy()
            if policy.mode == "isolated":
                boundaries.append(left - start)
                if overflow == "split":
                    # Sequence splitting precedes packing, even if lengths differ.
                    next_boundary = (a // sequence_length + 1) * sequence_length
                    boundaries.extend(range(doc_start + next_boundary - start, right - start, sequence_length))
    pack_offsets = np.append(np.arange(0, size, policy.max_length, dtype=np.int32), size).astype(np.int32)
    if policy.mode == "isolated":
        boundaries = np.union1d(boundaries, pack_offsets).astype(np.int64)
        labels[boundaries[:-1]] = -100
        segment_values = np.diff(boundaries)
        segment_offsets = np.searchsorted(boundaries, pack_offsets).astype(np.int32)
    else:
        segment_values = np.diff(pack_offsets).astype(np.int64)
        segment_offsets = np.arange(len(pack_offsets), dtype=np.int32)
    return pa.table({
        "input_ids": pa.ListArray.from_arrays(pack_offsets, pa.array(ids)),
        "labels": pa.ListArray.from_arrays(pack_offsets, pa.array(labels)),
        "segment_lengths": pa.ListArray.from_arrays(segment_offsets, pa.array(segment_values)),
    })


def pack_dense_arrow(dataset, policy, *, cache_dir=None, split_long=False, num_proc=1, overwrite_cache=False):
    """Exact dense packing, bounded token buffers per worker; cache independent of worker count."""
    if num_proc is None:
        num_proc = 1
    if not isinstance(num_proc, int) or isinstance(num_proc, bool) or num_proc < 1:
        raise ValueError("packing num_proc must be a positive integer")
    source = dataset.select_columns([c for c in ("input_ids", "labels") if c in dataset.column_names])
    fingerprint = Hasher.hash((ARROW_PACKING_VERSION, source._fingerprint, policy.to_dict(), split_long))
    root = Path(cache_dir or datasets_config.HF_DATASETS_CACHE) / "dense_packing" / fingerprint
    root.mkdir(parents=True, exist_ok=True)
    # A ready manifest avoids even the length scan at training startup. FileLock
    # serializes independent jobs sharing the cache, not just ranks of one job.
    with FileLock(str(root / "packing.lock")):
        ready = root / "ready.json"
        if ready.exists() and not overwrite_cache:
            files = json.loads(ready.read_text())
            if all((root / name).is_file() for name in files):
                return concatenate_datasets([Dataset.from_file(str(root / name)) for name in files])
        lengths = _lengths(source)
        sequence_length = policy.sequence_max_length or policy.max_length
        overflow = policy.overflow or ("split" if split_long and policy.mode == "isolated" else None)
        if overflow == "error" and np.any(lengths > sequence_length):
            index = int(np.flatnonzero(lengths > sequence_length)[0])
            raise ValueError(f"Example {index} exceeds sequence.max_length: {lengths[index]} > {sequence_length}")
        if overflow == "drop":
            lengths[lengths > sequence_length] = 0
        elif overflow == "truncate":
            lengths = np.minimum(lengths, sequence_length)
        offsets = np.concatenate(([0], np.cumsum(lengths)))
        tokens = int(offsets[-1])
        if policy.tail == "drop":
            tokens = tokens // policy.max_length * policy.max_length
        features = Features({name: Sequence(Value("int64")) for name in ("input_ids", "labels", "segment_lengths")})
        if not tokens:
            return Dataset.from_dict({name: [] for name in features}, features=features)
        task_tokens = policy.max_length * PACKS_PER_TASK
        starts = np.arange(0, tokens, task_tokens, dtype=np.int64)
        tasks = Dataset.from_dict({"start": starts, "stop": np.minimum(starts + task_tokens, tokens)})
        workers = min(num_proc, len(tasks))
        print(f"[packing] dense Arrow: {tokens:,} tokens → {(tokens + policy.max_length - 1) // policy.max_length:,} packs; "
              f"workers={workers}, up to {PACKS_PER_TASK} packs/task", flush=True)
        result = tasks.map(
            _pack_ranges, batched=True, batch_size=1, num_proc=workers,
            fn_kwargs=dict(source=source, offsets=offsets, policy=policy,
                           sequence_length=sequence_length, overflow=overflow),
            remove_columns=tasks.column_names, features=features,
            cache_file_name=str(root / "packs.arrow"), new_fingerprint=fingerprint,
            load_from_cache_file=not overwrite_cache,
            desc=f"Dense packing ({PACKS_PER_TASK} packs/task)",
        )
        files = [Path(entry["filename"]).name for entry in result.cache_files]
        temporary = root / "ready.json.tmp"
        temporary.write_text(json.dumps(files) + "\n")
        temporary.replace(ready)
        return result
