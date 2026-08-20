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
    text_only: bool = field(
        default=False,
        metadata={"help": "Load text-only class (e.g. Qwen3_5ForCausalLM) instead of "
                          "the multimodal wrapper; skips unused vision tower weights"},
    )
    fp8_storage: bool = field(
        default=False,
        metadata={"help": "QLoRA-style FP8 weight storage: load an FP8-quantized checkpoint, "
                          "keep frozen weights in fp8, dequantize to bf16 inside forward. "
                          "Halves base-weight memory so gradient checkpointing can be disabled. "
                          "Requires a checkpoint with quantization_config quant_method=fp8 "
                          "(HF finegrained block-wise)"},
    )
    fp8_compile_dequant: bool = field(
        default=False,
        metadata={"help": "torch.compile the fp8 dequant op (single fused memory pass). "
                          "Only with fp8_storage; reduces dequant overhead in forward/backward"},
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
class SFTConfig:
    """Config for SFT (Supervised Fine-Tuning) dataset pipeline.

    Special token IDs for boundary detection:
    - im_start_token_id: Token ID for sequence start marker (e.g. 291347 for Qwen3.5)
    - im_end_token_id: Token ID for sequence end marker (e.g. 291348 for Qwen3.5)
    - think_start_token_id: Token ID for think block start (e.g. 291370 for Qwen3.5)
    - think_end_token_id: Token ID for think block end (e.g. 291371 for Qwen3.5)
    - assistant_role_string: String whose first encoded token is used as assistant role ID
    - newline_token_id: Token ID for newline character (e.g. 198 for Qwen3.5)

    These must be specified in config because some special tokens (like im_start/im_end)
    are not reliably detectable via tokenizer.convert_tokens_to_ids() on all models.
    """

    max_tokens_count: int = field(
        default=2048,
        metadata={"help": "Max token length for SFT conversations"},
    )
    only_target_loss: bool = field(
        default=True,
        metadata={"help": "Mask non-assistant tokens in labels (only compute loss on assistant responses)"},
    )
    sample_rate: float = field(
        default=1.0,
        metadata={"help": "Fraction of data to use (0.0-1.0). Random subsampling."},
    )
    mask_think_block: bool = field(
        default=False,
        metadata={"help": "Mask empty THINK blocks in labels. Only masks when think content is whitespace."},
    )
    dynamic_padding: bool = field(
        default=True,
        metadata={"help": "Pad to max length in batch (True) vs global max_length (False)"},
    )
    pad_to_multiple_of: int = field(
        default=8,
        metadata={"help": "Pad sequence length to nearest multiple of this value"},
    )
    packing: bool = field(
        default=False,
        metadata={"help": "Strict document packing for train split: whole samples are "
                          "greedily packed into fixed-size chunks with per-document "
                          "position_ids reset, seq_idx and cu_seqlens (batch_size=1). "
                          "Eval split is never packed."},
    )
    pack_chunk_size: int = field(
        default=4096,
        metadata={"help": "Chunk length for packing. Must be >= max_tokens_count "
                          "(longer samples would be dropped)"},
    )
    im_start_token_id: Optional[int] = field(
        default=None,
        metadata={"help": "Token ID for sequence start marker. Required for assistant boundary detection."},
    )
    im_end_token_id: Optional[int] = field(
        default=None,
        metadata={"help": "Token ID for sequence end marker."},
    )
    think_start_token_id: Optional[int] = field(
        default=None,
        metadata={"help": "Token ID for think block start marker."},
    )
    think_end_token_id: Optional[int] = field(
        default=None,
        metadata={"help": "Token ID for think block end marker."},
    )
    assistant_role_string: str = field(
        default="assistant",
        metadata={"help": "String whose first encoded token is the assistant role ID in boundary pattern"},
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
    sft: Optional[SFTConfig] = field(
        default=None,
        metadata={"help": "SFT dataset config (chat-template tokenization, masking)"},
    )
    dataset_factory: Optional[str] = field(
        default=None,
        metadata={"help": "Dotpath to DatasetFactory class (e.g. my_module.MyFactory)"},
    )
    collator_factory: Optional[str] = field(
        default=None,
        metadata={"help": "Dotpath to CollatorFactory class"},
    )
