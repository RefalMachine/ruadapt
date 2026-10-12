"""Optimized packing must be byte-for-byte equivalent to the sequential reference."""
from dataclasses import replace

import pytest
from datasets import Dataset

from ruadapt.training.config.packing import PackingPolicy
from ruadapt.training.datasets.document_packing import iter_packs, pack_dataset


@pytest.mark.parametrize("mode", ["stream", "isolated"])
@pytest.mark.parametrize("overflow", [None, "split", "drop", "truncate"])
@pytest.mark.parametrize("tail", ["pad", "drop", "keep"])
def test_dense_equivalent_across_workers(tmp_path, monkeypatch, mode, overflow, tail):
    # Small tasks force boundaries through long docs and across empty/dropped docs.
    monkeypatch.setattr("ruadapt.training.datasets.arrow_packing.PACKS_PER_TASK", 2)
    rows = [{"input_ids": list(range(n)), "labels": [-100 if i % 3 == 0 else i + 100 for i in range(n)]}
            for n in [0, 1, 7, 0, 31, 2, 6, 50, 0, 4, 8, 1]]
    data = Dataset.from_list(rows)
    # Exercise Arrow slices/chunks and HF non-contiguous index mappings.
    data = data.select([4, 0, 5, 2, 7, 10, 1, 8, 11, 6, 3, 9])
    policy = PackingPolicy("sft", mode, 8, "tokens", "dense", tail, 6, overflow)
    expected = list(iter_packs(data, mode=mode, max_length=8, strategy="dense", tail=tail,
                               sequence_max_length=6, overflow=overflow))
    for workers in (1, 3):
        actual = pack_dataset(data, policy, cache_dir=str(tmp_path / str(workers)), num_proc=workers)
        assert actual.to_list() == expected


def test_labels_optional_and_cache_reused_with_other_workers(tmp_path, monkeypatch):
    data = Dataset.from_dict({"input_ids": [list(range(29)), [], [1, 2, 3]]})
    policy = PackingPolicy("cpt", "isolated", 8, "tokens", "dense", "keep")
    expected = list(iter_packs(data, mode="isolated", max_length=8, strategy="dense", split_long=True))
    first = pack_dataset(data, policy, cache_dir=str(tmp_path), split_long=True, num_proc=2)
    assert first.to_list() == expected
    def fail(*args, **kwargs):
        raise AssertionError("cache hit must not scan lengths or regenerate packs")
    monkeypatch.setattr("ruadapt.training.datasets.arrow_packing._lengths", fail)
    second = pack_dataset(data, policy, cache_dir=str(tmp_path), split_long=True, num_proc=4)
    assert second.to_list() == expected
    with pytest.raises(ValueError, match="Packing failed"):
        pack_dataset(data, policy, cache_dir=str(tmp_path), split_long=True, overwrite_cache=True)


def test_invalid_labels_and_overflow_error(tmp_path):
    policy = PackingPolicy("cpt", "isolated", 8, "tokens", "dense", "pad", 4, "error")
    for rows, cause in [([{"input_ids": [1, 2], "labels": [1]}], "same length"),
                        ([{"input_ids": [1] * 5}], "Example 0 exceeds")]:
        with pytest.raises(ValueError, match="Packing failed") as error:
            pack_dataset(Dataset.from_list(rows), policy, cache_dir=str(tmp_path))
        assert cause in str(error.value.__cause__)


def test_all_filtered_and_dropped_tail(tmp_path):
    data = Dataset.from_dict({"input_ids": [list(range(9))]})
    policy = PackingPolicy("cpt", "isolated", 8, "tokens", "dense", "drop", 4, "drop")
    assert len(pack_dataset(data, policy, cache_dir=str(tmp_path))) == 0
    assert len(pack_dataset(data, replace(policy, overflow="truncate"), cache_dir=str(tmp_path))) == 0
