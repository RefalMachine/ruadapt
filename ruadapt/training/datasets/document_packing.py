"""Backend-independent packing. Segment lengths are the sole source of boundaries.

The reference iterator carries partial packs across records/map batches.
Dense uses an equivalent parallel Arrow implementation; whole/none use
from_generator. Neither path collects the token corpus in RAM.
"""

from datasets import Dataset, Features, Sequence, Value

PACKING_SCHEMA_VERSION = 2


def iter_sequences(records, *, max_length, overflow):
    """Apply token limits to complete examples with their already constructed labels."""
    if max_length < 2 or overflow not in {"split", "drop", "truncate", "error"}:
        raise ValueError("Invalid sequence limit/overflow")
    for index, record in enumerate(records):
        x = list(record["input_ids"])
        y = list(record.get("labels", x))
        if len(x) != len(y):
            raise ValueError("input_ids and labels must have the same length")
        if not x:
            continue
        if len(x) > max_length:
            if overflow == "drop":
                continue
            if overflow == "error":
                raise ValueError(f"Example {index} exceeds sequence.max_length: {len(x)} > {max_length}")
            if overflow == "truncate":
                x, y = x[:max_length], y[:max_length]
        for offset in range(0, len(x), max_length):
            yield {"input_ids": x[offset:offset + max_length], "labels": y[offset:offset + max_length]}


def iter_packs(records, *, mode, max_length, split_long=False, strategy=None, tail="keep",
               sequence_max_length=None, overflow=None):
    """Pack sequentially. Dense fills every pack; whole preserves example fragments.

    Tail only controls emission of the last incomplete pack. Padding is applied
    by the collator, so canonical records contain only real input tokens.
    """
    if max_length < 2 or mode not in {"none", "stream", "isolated"}:
        raise ValueError("Packing requires a known mode and max_length >= 2")
    strategy = strategy or {"none": "none", "stream": "dense", "isolated": "whole"}[mode]
    if strategy not in {"none", "whole", "dense"} or tail not in {"keep", "pad", "drop"}:
        raise ValueError("Invalid packing strategy/tail")
    if overflow is not None:
        records = iter_sequences(records, max_length=sequence_max_length or max_length, overflow=overflow)
    elif split_long and mode == "isolated":
        records = iter_sequences(records, max_length=max_length, overflow="split")
    ids, labels, lengths = [], [], []
    for record in records:
        x, y = list(record["input_ids"]), list(record.get("labels", record["input_ids"]))
        if len(x) != len(y):
            raise ValueError("input_ids and labels must have the same length")
        if not x:
            continue
        if strategy == "none":
            yield {"input_ids": x, "labels": y, "segment_lengths": [len(x)]}
            continue
        if strategy == "whole":
            if len(x) > max_length:
                raise ValueError("A whole sample exceeds packing.max_length")
            if len(ids) + len(x) > max_length:
                yield {"input_ids": ids, "labels": labels,
                       "segment_lengths": lengths if mode == "isolated" else [len(ids)]}
                ids, labels, lengths = [], [], []
        offset = 0
        while offset < len(x):
            count = min(max_length - len(ids), len(x) - offset)
            sx, sy = x[offset:offset + count], y[offset:offset + count]
            if mode == "isolated":
                sy[0] = -100
            ids.extend(sx)
            labels.extend(sy)
            lengths.append(count)
            offset += count
            if len(ids) == max_length:
                yield {"input_ids": ids, "labels": labels,
                       "segment_lengths": lengths if mode == "isolated" else [len(ids)]}
                ids, labels, lengths = [], [], []
    if ids and tail != "drop":
        yield {"input_ids": ids, "labels": labels,
               "segment_lengths": lengths if mode == "isolated" else [len(ids)]}


def _generate_packs(dataset, policy, split_long, schema_version):
    yield from iter_packs(dataset, mode=policy.mode, max_length=policy.max_length, split_long=split_long,
                          strategy=policy.strategy, tail=policy.tail,
                          sequence_max_length=policy.sequence_max_length, overflow=policy.overflow)


def pack_dataset(dataset, policy, *, cache_dir=None, split_long=False, num_proc=1, overwrite_cache=False):
    features = Features({name: Sequence(Value("int64")) for name in ("input_ids", "labels", "segment_lengths")})
    if not len(dataset):
        return Dataset.from_dict({key: [] for key in features}, features=features)
    try:
        if policy.strategy == "dense":
            from ruadapt.training.datasets.arrow_packing import pack_dense_arrow
            return pack_dense_arrow(dataset, policy, cache_dir=cache_dir, split_long=split_long,
                                    num_proc=num_proc, overwrite_cache=overwrite_cache)
        return Dataset.from_generator(
            _generate_packs,
            gen_kwargs=dict(dataset=dataset, policy=policy, split_long=split_long,
                            schema_version=PACKING_SCHEMA_VERSION),
            features=features, cache_dir=cache_dir,
        )
    except Exception as exc:
        # Datasets may wrap errors from the iterator; retain the original cause.
        raise ValueError("Packing failed; check sequence limits, tail policy and whether any examples remain") from exc


def packing_stats(dataset, policy, *, freeze_idx=None):
    """Optional single streaming pass; memory is O(vocabulary), not O(corpus)."""
    from collections import Counter

    frequency = Counter()
    tokens = targets = substitutions = segments = 0
    for row in dataset:
        ids, labels = row["input_ids"], row["labels"]
        tokens += len(ids)
        targets += sum(y != -100 for y in labels[1:])
        substitutions += sum(x != y and y != -100 for x, y in zip(ids, labels))
        segments += len(row["segment_lengths"])
        frequency.update(x for x in ids if freeze_idx is None or x >= freeze_idx)
    return {"policy": policy.to_dict(), "schema_version": PACKING_SCHEMA_VERSION,
            "n_chunks": len(dataset), "n_segments": segments, "input_tokens": tokens,
            "target_tokens": targets, "target_substitutions": substitutions,
            "fill": (1.0 if tokens and policy.mode == "isolated" and policy.tail == "keep" else
                     tokens / max(len(dataset) * policy.max_length, 1)),
            "nominal_pack_fill": tokens / max(len(dataset) * policy.max_length, 1),
            "token_frequency": dict(frequency)}
