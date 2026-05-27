"""Unified config dataclasses for ruadapt.training.

All config is parsed from JSON via HfArgumentParser. MainConfig is the top-level
container; sub-configs (ModelConfig, DataConfig, etc.) can also be used independently.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from transformers import TrainingArguments


@dataclass
class ModelConfig:
    model_name_or_path: str = field(metadata={"help": "Path or HF Hub model id"})
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Tokenizer path if different from model"}
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Model dtype: bfloat16, float16, float32, auto"},
    )
    trust_remote_code: bool = False
    attn_implementation: str = field(
        default="flash_attention_2",
        metadata={"help": "Attention impl: flash_attention_2, sdpa, eager"},
    )


@dataclass
class DataConfig:
    train_file: str = field(metadata={"help": "Path to training data (JSONL/JSON/TXT)"})
    val_file: Optional[str] = field(
        default=None, metadata={"help": "Path to validation data"}
    )
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    block_size: Optional[int] = field(
        default=None,
        metadata={"help": "Sequence length after tokenization (for CLM)"},
    )
    preprocessing_num_workers: int = 8
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory for HF datasets Arrow cache. None = HF default (~/.cache/huggingface)"},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Force re-tokenization even if cache exists"},
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Use streaming mode for very large datasets (no disk cache)"},
    )
    max_text_length: Optional[int] = field(
        default=None,
        metadata={"help": "Filter documents longer than this (in characters). Recommended: 5000-50000"},
    )


@dataclass
class LoRAConfig:
    peft: bool = False
    r: int = 8
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Modules to make trainable (e.g. lm_head, embed_tokens)"},
    )


@dataclass
class FreezeConfig:
    strategy: str = field(
        default="none",
        metadata={"help": "Freeze strategy: none, embed_only, custom"},
    )
    freeze_idx: Optional[int] = field(
        default=None,
        metadata={"help": "Token IDs below this are frozen in embed (embed_only strategy)"},
    )
    unfreeze_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Module names to keep trainable (custom strategy)"},
    )


@dataclass
class TrainingConfig(TrainingArguments):
    """Extends HF TrainingArguments with ruadapt-training-specific fields."""

    embed_weight_decay: float = field(
        default=0.0,
        metadata={"help": "Separate weight decay for embeddings (default 0.0)"},
    )
    wsd_constant_part: float = field(
        default=0.85,
        metadata={"help": "Fraction of steps for constant LR phase in WSD scheduler"},
    )


@dataclass
class UnifiedDatasetConfig:
    """Config for PackedDataset (CPT/CLM/target substitution)."""

    natural_boundaries: bool = field(
        default=False,
        metadata={"help": "Cut chunks at BOS/EOS/newline boundaries (CLM mode)"},
    )
    fragment_ratio: float = field(
        default=0.0,
        metadata={"help": "Fraction of tokens to attempt BPE fragmentation (0.0-1.0)"},
    )
    p_split: float = field(
        default=0.3,
        metadata={"help": "Probability of splitting at each merge tree node"},
    )
    domain_filter_field: Optional[str] = field(
        default=None,
        metadata={"help": "JSON field to filter documents by"},
    )
    domain_filter_value: Optional[str] = field(
        default=None,
        metadata={"help": "Value to keep when filtering by domain_filter_field"},
    )
    domain_filter_exclude_value: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Values to exclude when filtering by domain_filter_field"},
    )
    compute_stats: bool = field(
        default=False,
        metadata={"help": "Compute token frequency, histograms, save stats JSON. "
                          "Useful once for fixed tokenization; skip when tuning bs/lr."},
    )
    substitution_method: str = field(
        default="none",
        metadata={"help": "Substitution method: none, random (BPE-Dropout), targeted, hybrid"},
    )
    trim_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to trim_tokenizer output (for targeted substitution)"},
    )
    min_parent_freq_ratio: float = field(
        default=3.0,
        metadata={"help": "Min parent freq as multiple of K for targeted substitution"},
    )
    random_sub_ratio: float = field(
        default=0.0,
        metadata={"help": "Probability multiplier for random phase in hybrid mode. "
                          "E.g. 0.1 with fragment_ratio=0.2 gives ~2% random substitution."},
    )


@dataclass
class MainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    freeze: FreezeConfig = field(default_factory=FreezeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    unified_dataset: Optional[UnifiedDatasetConfig] = field(
        default=None,
        metadata={"help": "Unified dataset config (CPT/CLM/target substitution)"},
    )
    dataset_factory: Optional[str] = field(
        default=None,
        metadata={"help": "Dotpath to DatasetFactory class (e.g. my_module.MyFactory)"},
    )
    collator_factory: Optional[str] = field(
        default=None,
        metadata={"help": "Dotpath to CollatorFactory class"},
    )
