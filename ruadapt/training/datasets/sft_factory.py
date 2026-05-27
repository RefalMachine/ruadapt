"""SFT factory example: chat-template tokenization with assistant boundary masking.

Based on reference_scripts/translation/train_sft_fsdp.py.
Demonstrates how to implement DatasetFactory + CollatorFactory for SFT.
"""

import functools
from typing import Any, Callable, Dict, List

from datasets import load_dataset
from torch.utils.data import Dataset

from ruadapt.training.datasets.collators import SimpleStackCollator
from ruadapt.training.datasets.in_memory import InMemoryPaddedDataset


def find_last_subsequence(tokens: List[int], pattern: List[int]) -> int:
    """Find the index after the last occurrence of pattern in tokens."""
    n, m = len(tokens), len(pattern)
    if m == 0 or n < m:
        return -1
    for i in range(n - m, -1, -1):
        if tokens[i : i + m] == pattern:
            return i + m
    return -1


def convert_single_record(
    example: Dict,
    *,
    tokenizer=None,
    max_tokens_count: int = 2048,
    only_target_loss: bool = True,
    labels_pad_token_id: int = -100,
    assistant_boundary_tokens: List[int] = None,
    assistant_boundary_tokens_no_think: List[int] = None,
) -> Dict:
    """Process a single SFT record: tokenize, mask prompt, return lists."""
    skip = {"input_ids": [], "labels": [], "attention_mask": [], "skip": True}

    try:
        input_ids = tokenizer.apply_chat_template(
            example["messages"],
            add_special_tokens=False,
            tokenize=True,
            add_generation_prompt=False,
        )
        if isinstance(input_ids, dict):
            input_ids = input_ids["input_ids"]
        elif hasattr(input_ids, "input_ids"):
            input_ids = input_ids.input_ids

        if not input_ids or len(input_ids) > max_tokens_count:
            return skip

        labels = list(input_ids)

        if only_target_loss:
            prompt_len = -1
            if assistant_boundary_tokens:
                prompt_len = find_last_subsequence(input_ids, assistant_boundary_tokens)
            if prompt_len == -1 and assistant_boundary_tokens_no_think:
                prompt_len = find_last_subsequence(
                    input_ids, assistant_boundary_tokens_no_think
                )
            if prompt_len == -1:
                return skip
            for i in range(prompt_len):
                labels[i] = labels_pad_token_id

        if all(l == labels_pad_token_id for l in labels):
            return skip

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
            "skip": False,
        }
    except Exception:
        return skip


class SFTDatasetFactory:
    """DatasetFactory for SFT with chat-template tokenization."""

    def create_train(self, tokenizer: Any, config: Any) -> Dataset:
        return self._create_dataset(tokenizer, config, split="train")

    def create_eval(self, tokenizer: Any, config: Any) -> Dataset:
        return self._create_dataset(tokenizer, config, split="validation")

    def _create_dataset(self, tokenizer, config, split: str) -> Dataset:
        import multiprocessing as mp

        data_config = config.data
        max_tokens_count = getattr(data_config, "block_size", 2048) or 2048
        only_target_loss = True

        # Load raw data
        data_file = data_config.train_file if split == "train" else data_config.val_file
        raw = load_dataset("json", data_files={split: data_file})[split]

        # Limit samples
        max_samples = (
            data_config.max_train_samples
            if split == "train"
            else data_config.max_val_samples
        )
        if max_samples is not None:
            n = min(max_samples, len(raw))
            raw = raw.select(range(n))

        # Precompute boundary tokens
        assistant_boundary_tokens = tokenizer.encode(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            add_special_tokens=False,
        )
        assistant_boundary_tokens_no_think = tokenizer.encode(
            "<|im_start|>assistant\n",
            add_special_tokens=False,
        )

        # Tokenize
        preprocess_fn = functools.partial(
            convert_single_record,
            tokenizer=tokenizer,
            max_tokens_count=max_tokens_count,
            only_target_loss=only_target_loss,
            labels_pad_token_id=-100,
            assistant_boundary_tokens=assistant_boundary_tokens,
            assistant_boundary_tokens_no_think=assistant_boundary_tokens_no_think,
        )

        num_proc = min(mp.cpu_count(), 8)
        processed = raw.map(
            preprocess_fn,
            batched=False,
            num_proc=num_proc,
            remove_columns=raw.column_names,
            load_from_cache_file=True,
        )
        processed = processed.filter(lambda x: not x["skip"], num_proc=num_proc)
        processed = processed.remove_columns(["skip"])

        # Convert to in-memory tensors
        return InMemoryPaddedDataset(
            processed, max_tokens_count, tokenizer.pad_token_id
        )


class SFTCollatorFactory:
    """CollatorFactory for SFT (stack pre-padded tensors)."""

    def create(self, tokenizer: Any, config: Any) -> Callable:
        return SimpleStackCollator()
