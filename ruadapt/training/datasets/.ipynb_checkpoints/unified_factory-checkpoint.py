"""Unified dataset factory: configurable pipeline for CPT, CLM, and target substitution.

Pipeline (each step cached independently via HF datasets cache):
    load_dataset → filter → ensure_tokenized → ensure_packed → PackedDataset

Config example:
{
  "dataset_factory": "ruadapt.training.datasets.unified_factory.UnifiedDatasetFactory",
  "collator_factory": "ruadapt.training.datasets.unified_factory.UnifiedCollatorFactory",
  "data": {
    "train_file": "data/train.json",
    "val_file": "data/val.json",
    "block_size": 512,
    "preprocessing_num_workers": 8,
    "cache_dir": "/workdir/cache/ruadapt",
    "overwrite_cache": false
  },
  "unified_dataset": {
    "natural_boundaries": false,
    "fragment_ratio": 0.0,
    "p_split": 0.3
  }
}
"""

import os
from typing import Any, Callable

from datasets import load_dataset
from torch.utils.data import Dataset

from ruadapt.training.datasets.collators import PackedCollatorWithMask
from ruadapt.training.datasets.packing import (
    PackedDataset,
    compute_newline_ids,
    make_pack_fn,
)
from ruadapt.training.datasets.utils import _get_tokenize_fn, _is_rank_zero, _log, ensure_tokenized


def ensure_packed(tokenized, pack_fn, num_proc, overwrite_cache, is_main_process):
    """Pack tokenized dataset via .map() with caching. Only main rank writes cache.

    Analogous to ensure_tokenized but for the packing + substitution step.
    Supports parallel num_proc >= 1 because the packing function uses batch index offsets
    to ensure perfect determinism in random splits across all workers and ranks.
    """
    import torch.distributed as dist

    column_names = list(tokenized.features)
    distributed = dist.is_initialized()

    if not is_main_process:
        if distributed:
            _log("Waiting at barrier for pack cache...", main_process_only=False)
            dist.barrier()
            _log("Pack cache ready, loading...", main_process_only=False)
        return tokenized.map(
            pack_fn,
            batched=True,
            with_indices=True,
            num_proc=num_proc,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Packing (from cache)",
        )

    print("1", flush=True)
    _log(f"Packing dataset with num_proc={num_proc}...")
    result = tokenized.map(
        pack_fn,
        batched=True,
        with_indices=True,
        num_proc=num_proc,
        remove_columns=column_names,
        load_from_cache_file=not overwrite_cache,
        desc="Packing",
    )

    if distributed:
        _log("[rank 0] Packing done, signaling barrier...")
        dist.barrier()

    return result


class UnifiedDatasetFactory:
    """DatasetFactory for unified packed dataset.

    Reads config.unified_dataset for:
    - natural_boundaries (bool, default False)
    - fragment_ratio (float, default 0.0)
    - p_split (float, default 0.3)
    - substitution_method (str, default "none") — none/random/targeted/hybrid
    - random_sub_ratio (float, default 0.0) — for hybrid mode
    - domain_filter_field (str, optional) — JSON field to filter by
    - domain_filter_value (str, optional) — value to keep

    Uses HF datasets (Arrow-backed) for loading, filtering, tokenization,
    and packing. Each step is cached independently via HF datasets cache.
    """

    def create_train(self, tokenizer: Any, config: Any) -> Dataset:
        return self._create_dataset(tokenizer, config, split="train")

    def create_eval(self, tokenizer: Any, config: Any) -> Dataset:
        return self._create_dataset(tokenizer, config, split="eval")

    def _create_dataset(self, tokenizer, config, split: str) -> Dataset:
        data_config = config.data
        block_size = data_config.block_size or 512

        # Unified dataset config (with defaults)
        unified_cfg = getattr(config, "unified_dataset", None)
        if unified_cfg is not None:
            natural_boundaries = getattr(unified_cfg, "natural_boundaries", False)
            fragment_ratio = getattr(unified_cfg, "fragment_ratio", 0.0)
            p_split = getattr(unified_cfg, "p_split", 0.3)
            domain_filter_field = getattr(unified_cfg, "domain_filter_field", None)
            domain_filter_value = getattr(unified_cfg, "domain_filter_value", None)
            domain_filter_exclude_value = getattr(unified_cfg, "domain_filter_exclude_value", None)
            substitution_method = getattr(unified_cfg, "substitution_method", "none")
            min_parent_freq_ratio = getattr(unified_cfg, "min_parent_freq_ratio", 3.0)
            trim_dir = getattr(unified_cfg, "trim_dir", None)
            random_sub_ratio = getattr(unified_cfg, "random_sub_ratio", 0.0)
        else:
            natural_boundaries = False
            fragment_ratio = 0.0
            p_split = 0.3
            domain_filter_field = None
            domain_filter_value = None
            domain_filter_exclude_value = None
            substitution_method = "none"
            min_parent_freq_ratio = 3.0
            trim_dir = None
            random_sub_ratio = 0.0

        # Override substitution for eval
        if split == "eval":
            fragment_ratio = 0.0
            substitution_method = "none"

        # Load data via HF datasets (Arrow-backed)
        data_file = data_config.train_file if split == "train" else data_config.val_file
        if data_file is None:
            return None

        extension = data_file.split(".")[-1]
        if extension in ("jsonl", "json"):
            extension = "json"

        cache_dir = getattr(data_config, "cache_dir", None)
        overwrite_cache = getattr(data_config, "overwrite_cache", False)
        num_proc = getattr(data_config, "preprocessing_num_workers", 8)

        # Limit samples via split slicing (avoids loading full file)
        max_samples = (
            data_config.max_train_samples
            if split == "train"
            else data_config.max_val_samples
        )
        if max_samples is not None:
            split_spec = f"{split}[:{max_samples}]"
        else:
            split_spec = split

        raw = load_dataset(
            extension,
            data_files={split: data_file},
            cache_dir=cache_dir,
            split=split_spec,
        )

        # Filter by text length (Arrow-backed)
        max_text_length = getattr(data_config, "max_text_length", None)
        if max_text_length is not None:
            raw = raw.filter(lambda x: len(x["text"]) <= max_text_length)

        # Domain filtering (Arrow-backed)
        if domain_filter_field and domain_filter_value is not None:
            raw = raw.filter(
                lambda x: x.get(domain_filter_field) == domain_filter_value
            )
        elif domain_filter_field and domain_filter_exclude_value is not None:
            raw = raw.filter(
                lambda x: x.get(domain_filter_field) not in domain_filter_exclude_value
            )

        # Step 1: Tokenize (parallel, cached, only main rank writes cache)
        tokenize_fn = _get_tokenize_fn(tokenizer)
        tokenized = ensure_tokenized(
            raw, tokenize_fn, num_proc,
            overwrite_cache=overwrite_cache,
            is_main_process=_is_rank_zero(),
        )

        # Step 2: Pack + substitute (cached, single-process for determinism)
        seed = getattr(config.training, "seed", 42)
        newline_ids = compute_newline_ids(tokenizer) if natural_boundaries else set()

        # K for targeted substitution: use freeze_idx-based heuristic or default
        freeze_idx_val = getattr(config.freeze, "freeze_idx", 248044) or 248044
        K = 50  # Default threshold for rare tokens

        pack_fn = make_pack_fn(
            tokenizer=tokenizer,
            max_length=block_size,
            natural_boundaries=natural_boundaries,
            fragment_ratio=fragment_ratio,
            p_split=p_split,
            seed=seed,
            newline_ids=newline_ids,
            substitution_method=substitution_method,
            min_parent_freq_ratio=min_parent_freq_ratio,
            K=K,
            trim_dir=trim_dir,
            random_sub_ratio=random_sub_ratio,
        )

        print("2", flush=True)
        packed = ensure_packed(
            tokenized, pack_fn, num_proc=num_proc,
            overwrite_cache=overwrite_cache,
            is_main_process=_is_rank_zero(),
        )

        freeze_idx = getattr(config.freeze, "freeze_idx", None)
        compute_stats = getattr(unified_cfg, "compute_stats", False) if unified_cfg else False

        dataset = PackedDataset(
            pre_packed_dataset=packed,
            tokenizer=tokenizer,
            max_length=block_size,
            natural_boundaries=natural_boundaries,
            fragment_ratio=fragment_ratio,
            p_split=p_split,
            freeze_idx=freeze_idx,
            n_documents=len(tokenized),
            compute_token_freq=compute_stats,
        )

        if _is_rank_zero():
            n_chunks = dataset.stats.get("n_chunks", 0)
            total_tokens = dataset.stats.get("total_tokens", 0)
            usable_tokens = dataset.stats.get("usable_tokens", 0)
            print(f"[{split}] {n_chunks:,} chunks × {block_size} = {usable_tokens:,} usable tokens "
                  f"(from {total_tokens:,} total, {dataset.stats.get('n_documents', 0):,} docs)")

        # Stats output (rank 0 only, train only, only if compute_stats=True)
        if split == "train" and _is_rank_zero() and compute_stats:
            from ruadapt.training.datasets.stats import (
                print_dataset_stats,
                plot_fragment_histogram,
                plot_token_frequency_histogram,
                save_dataset_stats,
            )

            output_dir = config.training.output_dir
            stats_dir = os.path.join(output_dir, "dataset_stats")

            print_dataset_stats(dataset, tokenizer, top_k=20)
            save_dataset_stats(
                dataset.stats, os.path.join(stats_dir, f"stats_{split}.json")
            )
            if dataset.stats.get("fragment_length_distribution"):
                plot_fragment_histogram(
                    dataset.stats,
                    save_path=os.path.join(stats_dir, f"frag_histogram_{split}.png"),
                )
            plot_token_frequency_histogram(
                dataset.stats,
                tokenizer=tokenizer,
                save_path=os.path.join(stats_dir, f"token_freq_histogram_{split}.png"),
            )

        return dataset


class UnifiedCollatorFactory:
    """CollatorFactory for unified packed dataset.

    Uses PackedCollatorWithMask which handles both modes:
    - When labels == input_ids (CPT/CLM): labels are cloned from input_ids
    - When labels != input_ids (target substitution): uses pre-computed labels
    """

    def create(self, tokenizer: Any, config: Any) -> Callable:
        return PackedCollatorWithMask()
