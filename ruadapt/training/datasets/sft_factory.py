"""SFT dataset factory: composable pipeline for chat-template SFT.

Pipeline steps (each independently testable):
    filter -> tokenize -> mask -> pack (optional, only for static padding)

Based on deprecated/instruct_tuning/train_sft_fsdp.py, substantially reworked.

Usage in config:
{
    "dataset_factory": "ruadapt.training.datasets.sft_factory.SFTDatasetFactory",
    "collator_factory": "ruadapt.training.datasets.sft_factory.SFTCollatorFactory",
    "sft": {
        "max_tokens_count": 2048,
        "only_target_loss": true,
        "sample_rate": 1.0,
        "mask_think_block": false,
        "dynamic_padding": true,
        "pad_to_multiple_of": 8
    }
}
"""

import functools
import multiprocessing as mp
import random
from typing import Any, Callable, Dict, List, Optional

from datasets import Dataset, load_dataset
from torch.utils.data import Dataset as TorchDataset

from ruadapt.training.datasets.collators import DynamicPadCollator, SimpleStackCollator
from ruadapt.training.datasets.in_memory import InMemoryPaddedDataset


# ---------------------------------------------------------------------------
# Helper: find all occurrences of a subsequence in a list
# ---------------------------------------------------------------------------

def find_all_subsequences(seq: List[int], pattern: List[int]) -> List[int]:
    """Find all start positions of pattern in seq."""
    n, m = len(seq), len(pattern)
    if m == 0 or n < m:
        return []
    return [i for i in range(n - m + 1) if seq[i : i + m] == pattern]


def find_last_subsequence(seq: List[int], pattern: List[int]) -> int:
    """Find the index after the last occurrence of pattern in seq. Returns -1 if not found."""
    n, m = len(seq), len(pattern)
    if m == 0 or n < m:
        return -1
    for i in range(n - m, -1, -1):
        if seq[i : i + m] == pattern:
            return i + m
    return -1


# ---------------------------------------------------------------------------
# Step 1: Filter
# ---------------------------------------------------------------------------

def filter_record(
    example: Dict,
    max_tokens_count: int = 2048,
    sample_rate: float = 1.0,
) -> bool:
    """Check if record should be included. Returns True to keep, False to skip.

    Checks:
    - messages field exists and is non-empty
    - sample_rate random subsampling
    - rough length heuristic (char count)
    """
    messages = example.get("messages")
    if not messages or not isinstance(messages, list) or len(messages) == 0:
        return False

    if sample_rate < 1.0 and random.random() > sample_rate:
        return False

    # Rough length heuristic: estimate ~3 chars per token
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    if total_chars > max_tokens_count * 4:
        return False

    return True


# ---------------------------------------------------------------------------
# Step 2: Tokenize
# ---------------------------------------------------------------------------

def tokenize_record(
    example: Dict,
    tokenizer: Any = None,
    max_tokens_count: int = 2048,
) -> Dict:
    """Tokenize full conversation via apply_chat_template.

    Returns dict with input_ids and skip flag.
    """
    skip = {"input_ids": [], "skip": True}

    try:
        result = tokenizer.apply_chat_template(
            example["messages"],
            add_special_tokens=False,
            tokenize=True,
            add_generation_prompt=False,
        )

        # Handle different return types
        if isinstance(result, dict):
            input_ids = result.get("input_ids", [])
        elif hasattr(result, "input_ids"):
            input_ids = result.input_ids
        elif hasattr(result, "data") and isinstance(result.data, dict):
            # BatchEncoding (UserDict subclass)
            input_ids = result.data.get("input_ids", [])
        else:
            input_ids = result

        if not input_ids or len(input_ids) > max_tokens_count:
            return skip

        return {"input_ids": input_ids, "skip": False}

    except Exception:
        return skip


# ---------------------------------------------------------------------------
# Step 3: Mask
# ---------------------------------------------------------------------------


def mask_record(
    example: Dict,
    *,
    im_start_token_id: int,
    im_end_token_id: int,
    assistant_role_id: int,
    newline_token_id: int,
    only_target_loss: bool = True,
    mask_think_block: bool = False,
    think_start_token_id: Optional[int] = None,
    think_end_token_id: Optional[int] = None,
    labels_pad_token_id: int = -100,
) -> Dict:
    """Create labels by masking non-assistant tokens.

    Strategy:
    1. Copy input_ids to labels
    2. Find assistant boundaries: [im_start, assistant_role_id, newline]
    3. If only_target_loss: mask everything outside assistant content spans
    4. If mask_think_block: mask empty think blocks within assistant content

    All token IDs must be provided via config — no auto-detection.

    Returns dict with input_ids, labels, skip flag.
    """
    input_ids = example.get("input_ids", [])
    if not input_ids:
        return {"input_ids": [], "labels": [], "attention_mask": [], "skip": True}

    labels = list(input_ids)

    if only_target_loss:
        # Build boundary pattern: [IM_START, assistant_role, NEWLINE]
        boundary = [im_start_token_id, assistant_role_id, newline_token_id]
        boundary_positions = find_all_subsequences(input_ids, boundary)

        if not boundary_positions:
            # No assistant response found — skip
            return {"input_ids": [], "labels": [], "attention_mask": [], "skip": True}

        # Determine content spans for each assistant response
        assistant_spans = []
        for pos in boundary_positions:
            content_start = pos + len(boundary)
            # Find end: next IM_START or end of sequence
            content_end = len(input_ids)
            for j in range(content_start + 1, len(input_ids)):
                if input_ids[j] == im_start_token_id:
                    content_end = j
                    break
            assistant_spans.append((content_start, content_end))

        # Mask everything that is NOT inside an assistant content span
        for i in range(len(input_ids)):
            is_in_assistant = any(start <= i < end for start, end in assistant_spans)
            if not is_in_assistant:
                labels[i] = labels_pad_token_id

        # Optionally mask empty think blocks within assistant content
        if mask_think_block and think_start_token_id is not None and think_end_token_id is not None:
            for span_start, span_end in assistant_spans:
                _mask_empty_think_blocks(
                    input_ids, labels, span_start, span_end,
                    think_start_token_id, think_end_token_id, newline_token_id,
                    labels_pad_token_id,
                )

    # Check that there are some non-masked labels
    if all(l == labels_pad_token_id for l in labels):
        return {"input_ids": [], "labels": [], "attention_mask": [], "skip": True}

    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids), "skip": False}


def _mask_empty_think_blocks(
    input_ids: List[int],
    labels: List[int],
    span_start: int,
    span_end: int,
    think_start_id: int,
    think_end_id: int,
    newline_token_id: int,
    labels_pad_token_id: int = -100,
) -> None:
    """Mask empty think blocks within an assistant content span.

    A think block is considered "empty" if all tokens between THINK_START and
    THINK_END are whitespace (any token that decodes to only whitespace characters).

    Modifies labels in-place.
    """
    i = span_start
    while i < span_end:
        if input_ids[i] == think_start_id:
            # Find matching THINK_END
            think_end_pos = None
            for j in range(i + 1, span_end):
                if input_ids[j] == think_end_id:
                    think_end_pos = j
                    break

            if think_end_pos is not None:
                # Check if content between THINK_START and THINK_END is only whitespace.
                # We check the raw token IDs for common whitespace tokens:
                # single newline, double newline, space, tab, carriage return.
                content_between = input_ids[i + 1 : think_end_pos]
                whitespace_ids = {
                    newline_token_id,  # single newline
                    271,               # double newline (\n\n) - common in Qwen3.5
                    32,                # space
                    9,                 # tab
                    13,                # carriage return
                }
                is_empty = all(t in whitespace_ids for t in content_between)

                if is_empty:
                    # Mask THINK_START + content + THINK_END
                    for k in range(i, think_end_pos + 1):
                        labels[k] = labels_pad_token_id
                    # Also mask trailing whitespace tokens after THINK_END
                    k = think_end_pos + 1
                    while k < span_end and input_ids[k] in whitespace_ids:
                        labels[k] = labels_pad_token_id
                        k += 1

                i = think_end_pos + 1
            else:
                i += 1
        else:
            i += 1


# ---------------------------------------------------------------------------
# Step 4: Pack (only for static padding mode)
# ---------------------------------------------------------------------------

def pack_record(
    example: Dict,
    max_length: int = 2048,
    pad_token_id: int = 0,
    label_pad_token_id: int = -100,
    pad_to_multiple_of: int = 8,
) -> Dict:
    """Truncate, check for useful tokens, pad to fixed length.

    Returns skip=True if no useful (non-masked) tokens remain after truncation.
    """
    input_ids = example.get("input_ids", [])
    labels = example.get("labels", [])

    if not input_ids or not labels:
        return {"input_ids": [], "labels": [], "attention_mask": [], "skip": True}

    # Truncate
    input_ids = input_ids[:max_length]
    labels = labels[:max_length]

    # Check for useful tokens
    if all(l == label_pad_token_id for l in labels):
        return {"input_ids": [], "labels": [], "attention_mask": [], "skip": True}

    # Pad to target length
    target_len = max_length
    if pad_to_multiple_of and pad_to_multiple_of > 1:
        remainder = target_len % pad_to_multiple_of
        if remainder != 0:
            target_len += pad_to_multiple_of - remainder

    pad_len = target_len - len(input_ids)
    if pad_len > 0:
        input_ids = input_ids + [pad_token_id] * pad_len
        labels = labels + [label_pad_token_id] * pad_len

    attention_mask = [1] * (target_len - pad_len) + [0] * pad_len

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "skip": False,
    }


# ---------------------------------------------------------------------------
# Factory classes
# ---------------------------------------------------------------------------

def _resolve_token_id(tokenizer, value, token_str: str) -> int:
    """Resolve a token ID: use explicit value if provided, else try tokenizer.

    Args:
        tokenizer: HuggingFace tokenizer
        value: Explicit token ID from config (may be None)
        token_str: Token string to look up as fallback

    Returns:
        Token ID

    Raises:
        ValueError: If neither value nor tokenizer can provide the ID
    """
    if value is not None:
        return value
    token_id = tokenizer.convert_tokens_to_ids(token_str)
    if token_id is not None and token_id != tokenizer.unk_token_id:
        return token_id
    raise ValueError(
        f"Cannot resolve token ID for '{token_str}'. "
        f"Provide it explicitly in config (sft section)."
    )


class SFTDatasetFactory:
    """DatasetFactory for SFT with composable pipeline.

    Pipeline: load -> filter -> tokenize -> mask -> (optional) pack

    When dynamic_padding=True (default): returns HF dataset directly,
    collator handles padding at batch time.

    When dynamic_padding=False: adds pack step and returns InMemoryPaddedDataset.
    """

    def create_train(self, tokenizer: Any, config: Any) -> TorchDataset:
        return self._create_dataset(tokenizer, config, split="train")

    def create_eval(self, tokenizer: Any, config: Any) -> Optional[TorchDataset]:
        data_config = config.data
        if not data_config.val_file:
            return None
        return self._create_dataset(tokenizer, config, split="validation")

    def _create_dataset(self, tokenizer: Any, config: Any, split: str) -> TorchDataset:
        data_config = config.data
        sft_config = getattr(config, "sft", None)

        # Extract SFT parameters with fallbacks
        max_tokens_count = 2048
        only_target_loss = True
        sample_rate = 1.0
        mask_think_block = False
        dynamic_padding = True
        pad_to_multiple_of = 8

        if sft_config is not None:
            max_tokens_count = sft_config.max_tokens_count
            only_target_loss = sft_config.only_target_loss
            sample_rate = sft_config.sample_rate
            mask_think_block = sft_config.mask_think_block
            dynamic_padding = sft_config.dynamic_padding
            pad_to_multiple_of = sft_config.pad_to_multiple_of
        elif data_config.block_size:
            max_tokens_count = data_config.block_size

        # Resolve token IDs for boundary detection
        if sft_config is not None:
            im_start_id = _resolve_token_id(tokenizer, sft_config.im_start_token_id, chr(60) + chr(33) + "im_start" + chr(62))
            im_end_id = _resolve_token_id(tokenizer, sft_config.im_end_token_id, chr(60) + chr(33) + "im_end" + chr(62))

            # Newline token — always obtainable from tokenizer
            newline_ids = tokenizer.encode("\n", add_special_tokens=False)
            if not newline_ids:
                raise ValueError("Cannot encode newline character")
            newline_id = newline_ids[0]

            # Get assistant role ID from the role string
            assistant_role_str = sft_config.assistant_role_string
            assistant_ids = tokenizer.encode(assistant_role_str, add_special_tokens=False)
            if not assistant_ids:
                raise ValueError(f"Cannot encode assistant role string '{assistant_role_str}'")
            assistant_role_id = assistant_ids[0]

            think_start_id = sft_config.think_start_token_id
            think_end_id = sft_config.think_end_token_id
        else:
            # No SFT config — cannot proceed with masking
            raise ValueError(
                "SFT config (sft section) is required for SFTDatasetFactory. "
                "At minimum, provide im_start_token_id and im_end_token_id."
            )

        # Load raw data
        data_file = data_config.train_file if split == "train" else data_config.val_file
        raw = load_dataset("json", data_files={split: data_file}, split=split)

        # Limit samples
        max_samples = (
            data_config.max_train_samples if split == "train"
            else data_config.max_val_samples
        )
        if max_samples is not None:
            n = min(max_samples, len(raw))
            raw = raw.select(range(n))

        num_proc = min(mp.cpu_count(), 8)

        # Step 1: Filter
        filter_fn = functools.partial(
            filter_record,
            max_tokens_count=max_tokens_count,
            sample_rate=sample_rate,
        )
        filtered = raw.filter(filter_fn, num_proc=num_proc)

        # Step 2: Tokenize
        tokenize_fn = functools.partial(
            tokenize_record,
            tokenizer=tokenizer,
            max_tokens_count=max_tokens_count,
        )
        tokenized = filtered.map(
            tokenize_fn,
            batched=False,
            num_proc=num_proc,
            remove_columns=filtered.column_names,
            load_from_cache_file=True,
        )
        tokenized = tokenized.filter(lambda x: not x["skip"], num_proc=num_proc)
        tokenized = tokenized.remove_columns(["skip"])

        # Step 3: Mask
        mask_fn = functools.partial(
            mask_record,
            im_start_token_id=im_start_id,
            im_end_token_id=im_end_id,
            assistant_role_id=assistant_role_id,
            newline_token_id=newline_id,
            only_target_loss=only_target_loss,
            mask_think_block=mask_think_block,
            think_start_token_id=think_start_id,
            think_end_token_id=think_end_id,
        )
        masked = tokenized.map(
            mask_fn,
            batched=False,
            num_proc=num_proc,
            load_from_cache_file=True,
        )
        masked = masked.filter(lambda x: not x["skip"], num_proc=num_proc)
        masked = masked.remove_columns(["skip"])

        # Step 4: Pack (only for static padding)
        if not dynamic_padding:
            pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
            pack_fn = functools.partial(
                pack_record,
                max_length=max_tokens_count,
                pad_token_id=pad_token_id,
                pad_to_multiple_of=pad_to_multiple_of,
            )
            packed = masked.map(
                pack_fn,
                batched=False,
                num_proc=num_proc,
                load_from_cache_file=True,
            )
            packed = packed.filter(lambda x: not x["skip"], num_proc=num_proc)
            packed = packed.remove_columns(["skip"])

            return InMemoryPaddedDataset(packed, max_tokens_count, pad_token_id)

        # Dynamic padding: return HF dataset, collator pads at batch time
        return masked


class SFTCollatorFactory:
    """CollatorFactory for SFT.

    When dynamic_padding=True: returns DynamicPadCollator (pads to max in batch).
    When dynamic_padding=False: returns SimpleStackCollator (pre-padded tensors).
    """

    def create(self, tokenizer: Any, config: Any) -> Callable:
        sft_config = getattr(config, "sft", None)

        dynamic_padding = True
        pad_to_multiple_of = 8

        if sft_config is not None:
            dynamic_padding = sft_config.dynamic_padding
            pad_to_multiple_of = sft_config.pad_to_multiple_of

        if dynamic_padding:
            pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
            return DynamicPadCollator(
                pad_token_id=pad_token_id,
                pad_to_multiple_of=pad_to_multiple_of,
            )
        else:
            return SimpleStackCollator()
