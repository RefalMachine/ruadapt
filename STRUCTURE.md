# Project Structure

## Target Directory Tree

> **Status**: Migration complete. This is the current structure.
> See [MIGRATION_AND_REFACTORING_PLAN.md](MIGRATION_AND_REFACTORING_PLAN.md) for migration history.

```
devel/ruadapt/
├── pyproject.toml                          # Package metadata, dependencies, entry points
├── README.md                               # Project overview
├── AGENTS.md                               # Agent working rules and project context
├── STRUCTURE.md                            # This file
├── MIGRATION_AND_REFACTORING_PLAN.md       # Migration status and plan
├── .gitignore
├── .gitmodules                             # llmtf_open submodule config
│
├── ruadapt/                                # Main installable package
│   ├── __init__.py                         # __version__
│   │
    │   ├── tokenization/                       # Tokenizer manipulation + BPE tree
    │   │   ├── __init__.py
    │   │   ├── core.py                         # BPE learning, vocab injection (from extend_tokenizer.py)
    │   │   ├── replace.py                      # Embedding reinit: mean, wmean, random (from replace_tokenizer.py)
    │   │   ├── replace_batched.py              # Batched reinit with MLP head support
    │   │   ├── shrink.py                       # Vocab truncation (from shrink_tokenizer.py)
    │   │   ├── merges.py                       # BPE merge learning, tiktoken format (from add_merges_fast.py)
    │   │   ├── convert.py                      # Format converters: tiktoken↔HF, vocab extraction
    │   │   ├── bpe_tree.py                     # BPE merge tree — shared with training.datasets
    │   │   ├── utils.py                        # Token conversion, tokenizer properties, new-token helpers
    │   │   ├── cli.py                          # CLI wrappers (from run_replace_tokenizer.py)
    │   │   ├── trim.py                         # Cascade-trim rare terminal tokens from vocab
    │   │   ├── trim_model.py                   # Resize model embeddings after trimming
    │   │   ├── train_spm.py                    # SentencePiece training wrapper
│   │   └── evaluation/                     # Tokenizer quality evaluation
│   │       ├── evaluate_tokenizer.py       # Chars-per-token across languages
│   │       ├── compare_tokenizers.py       # Side-by-side tokenizer comparison
│   │       ├── run_diagnostic.py           # Full diagnostic (vocab, CPT, PPL)
│   │       └── data/                       # Sample data for evaluation (12 languages)
│   │
│   ├── initialization/                     # Embedding initialization research
│   │   ├── __init__.py
│   │   │
│   │   ├── data/                           # Data preparation
│   │   │   ├── __init__.py
│   │   │   ├── build_bpe_dataset.py        # BPE fragmentation dataset (train/val split)
│   │   │   ├── build_smart_dataset.py      # K-coverage dataset (dense, every token K+ times)
│   │   │   ├── build_smart_streaming.py    # K-coverage streaming variant
│   │   │   ├── build_seq_dataset.py        # Sequential dataset (for CPT)
│   │   │   ├── build_token_passport.py     # Token metadata (BPE depth, frequency, type)
│   │   │   ├── streamer.py                 # HF streaming wrapper with filtering
│   │   │   └── download.py                 # Download CPT evaluation data
│   │   │
│   │   ├── cache/                          # Hidden state caching
│   │   │   ├── __init__.py
│   │   │   └── precompute.py               # Precompute LLM hidden states (~2-3h, one-time)
│   │   │
│   │   ├── head/                           # MLP head training
│   │   │   ├── __init__.py
│   │   │   ├── train.py                    # Train head from cache (~30s/epoch)
│   │   │   └── metrics.py                  # FastMetrics: loss functions + eval metrics
│   │   │
│   │   ├── eval/                           # Initialization evaluation
│   │   │   ├── __init__.py
│   │   │   ├── per_token.py                # Per-token metrics on validation set
│   │   │   ├── init_quality.py             # Direct embedding comparison (mean vs head)
│   │   │   ├── delta_w.py                  # Delta-W diagnostic (pre/post CPT)
│   │   │   ├── micro_cpt.py                # Mini-CPT evaluation (golden metric)
│   │   │   ├── ppl.py                      # Standalone PPL evaluation
│   │   │   ├── baselines.py                # Mathematical baselines (mean, wmean)
│   │   │   └── diagnostic.py               # Full diagnostic pipeline
│   │   │
│   │   └── analytics/                      # Analysis and visualization
│   │       ├── __init__.py
│   │       ├── train_results.py            # Compare training runs (Markdown table)
│   │       ├── token_lifecycle.py          # Stratified lifecycle analysis
│   │       └── plot_logs.py                # Loss/metric curve plotting
│   │
│   ├── training/                           # Unified training core (from dgx_llm)
│   │   ├── __init__.py
│   │   ├── train.py                        # CLI entrypoint: python -m ruadapt.training.train
│   │   │
│   │   ├── config/                         # Configuration
│   │   │   ├── __init__.py
│   │   │   └── schema.py                   # HfArgumentParser dataclasses
│   │   │                                     #   ModelConfig, DataConfig, LoRAConfig,
│   │   │                                     #   FreezeConfig, TrainingConfig, MainConfig
│   │   │
│   │   ├── core/                           # Training infrastructure
│   │   │   ├── __init__.py
│   │   │   ├── model.py                    # load_model_and_tokenizer, apply_lora
│   │   │   ├── distributed.py              # FSDP/DDP/DeepSpeed init and wrapping
│   │   │   ├── trainer.py                  # UnifiedTrainer, WSD scheduler, callbacks
│   │   │   └── freeze.py                   # Freeze/unfreeze utilities, embed hooks
│   │   │
│   │   └── datasets/                       # Dataset handling
│   │       ├── __init__.py
│   │       ├── factory.py                  # DatasetFactory, CollatorFactory protocols
│   │       ├── utils.py                    # Tokenization helpers, caching
│   │       ├── packing.py                  # PackedDataset, BPE-Dropout substitution
│   │       ├── collators.py                # PackedCollatorWithMask, SimpleStackCollator
│   │       ├── in_memory.py                # InMemoryPaddedDataset (for SFT)
│   │       ├── unified_factory.py          # CPT/CLM/Substitution factory
│   │       ├── sft_factory.py              # SFT factory (chat template)
│   │       ├── stats.py                    # Dataset statistics, histograms
│   │       └── debug.py                    # print_sample diagnostic
│   │
│   ├── ushanka/                            # LEP — Layer Embedding Projection
│   │   ├── __init__.py
│   │   ├── compose.py                      # LEP orchestrator (from compose_ushanka.py)
│   │   ├── merge.py                        # make_ushanka() — model merging logic (from src/ushanka.py)
│   │   ├── projection.py                   # Projection modes (from src/ushanka_proj_utils.py)
│   │   ├── configs/                        # LEP configs per model architecture
│   │   └── templates/                      # Custom chat template JSONs
│   │
│   ├── evaluation/                         # Model evaluation
│   │   ├── __init__.py
│   │   ├── llmtf_open/                     # Git submodule: RefalMachine/llmtf_open
│   │   └── runner.py                       # Wrapper for running evaluations
│   │
│   ├── inference/                          # Batch inference
│   │   ├── __init__.py
│   │   ├── vllm.py                         # vLLM-based batch generation (from infer_vllm.py)
│   │   └── utils.py                        # Re-exports from ruadapt.utils.io
│   │
│   └── utils/                              # Shared utilities
│       ├── __init__.py
│       ├── io.py                           # read_jsonl, write_jsonl, read_json
│       ├── seed.py                         # set_random_seed
│       └── model_utils.py                  # load_causal_lm, resolve_model_class (consolidated)
│
├── deprecated/                             # Old code (reference only, not for import)
│   ├── README.md                           # What's here and why
│   ├── instruct_tuning/                    # Old instruct tuning (Unsloth-based)
│   │   ├── train_sft.py
│   │   ├── train_sft_transformers.py       # Reference for SFT factory
│   │   ├── train_sft_fsdp.py               # Reference for FSDP patterns
│   │   ├── train_dpo.py / train_kto.py / train_cpo.py / train_smpo.py
│   │   ├── train_cpo_transformers.py / train_kto_transformers.py / train_smpo_transformers.py
│   │   ├── train_cpo_fsdp.py
│   │   ├── train_unsloth.py
│   │   ├── smpo_trainer.py                 # ** Valuable ** — port to training/trainers/smpo.py when needed
│   │   ├── dataset.py / dpo_dataset.py
│   │   ├── utils.py
│   │   ├── models_configs/                 # Per-model training configs
│   │   └── datasets/                       # Dataset composition scripts
│   ├── pretraining/                        # Old HF Trainer pretraining
│   │   ├── train_trainer.py
│   │   └── utils.py
│   ├── root_scripts/                       # Old pipeline orchestrators
│   │   ├── run_pipeline.py
│   │   ├── run_pipeline_config.py
│   │   ├── run_pipeline_app.py
│   │   ├── run_pipeline_infer.py
│   │   ├── extend_or_replace.py
│   │   ├── mix_data.py
│   │   ├── run_pipeline.sh
│   │   ├── run_extend_or_replace.sh
│   │   ├── run_extend_or_replace_unigram.sh
│   │   ├── run_extension_pipeline.sh
│   │   ├── run_extension_pipeline_example.sh
│   │   ├── process_checkpoints.sh
│   │   └── process_loras.sh
│   └── pipeline_configs/                   # Old pipeline step configs (19 JSONs)
│
├── configs/                                # Training configs (dgx_llm JSON format)
│   ├── cpt/                                # CPT configs
│   ├── sft/                                # SFT configs
│   └── smoke/                              # Smoke test configs (fast, small)
│
├── deepspeed_configs/                      # DeepSpeed ZeRO configs
│   └── ds_z1_config.json
│
    ├── scripts/                                # Utility scripts
    │   ├── trim_tokenizer.py                   # Thin wrapper → ruadapt.tokenization.trim
    │   ├── trim_model.py                       # Thin wrapper → ruadapt.tokenization.trim_model
│   ├── analyze_targeted_substitution.py    # Substitution analysis (from dgx_llm)
│   ├── fix_config.py                       # Fix adapted model config for VLM eval (Qwen3.5)
│   ├── fix_configs.sh                      # Batch fix_config runner
│   ├── merge_lora.py                       # LoRA adapter merging
│   ├── merge_multiple_lora.py              # Multiple LoRA merging
│   ├── reinit_from_base.py                 # Reinitialize from base model
│   ├── convert2hf.py                       # Format conversion utility
│   ├── test_model_reinit.py                # Model reinit test utility
│   └── test_train_tokenization.py          # Train tokenization test utility
│
    ├── tests/                                  # Test suite (174 tests total)
    │   ├── conftest.py                         # Shared fixtures (small tokenizer, sample texts)
    │   ├── tokenization/
    │   │   ├── test_core.py                    # BPE learning, vocab injection
    │   │   ├── test_replace.py                 # Embedding reinit
    │   │   ├── test_merges.py                  # tiktoken merge learning
    │   │   ├── test_utils.py                   # Token conversion, properties
    │   │   ├── test_bpe_tree.py                # BPE merge tree
    │   │   └── test_convert.py                 # Format converters
    │   ├── initialization/
    │   │   └── test_initialization.py          # FastMetrics, LayerAttentionHead, BPE tree, utils (56 tests)
    │   ├── training/
    │   │   ├── conftest.py                     # Training fixtures
    │   │   ├── test_config.py                  # Config parsing
    │   │   ├── test_utils.py                   # Dataset utilities
    │   │   ├── test_bpe_tree.py                # BPE tree (import from tokenization)
    │   │   ├── test_collators.py               # Collators
    │   │   ├── test_unified_dataset.py         # Packed dataset
    │   │   ├── test_factory.py                 # Factory protocol
    │   │   ├── test_stats.py                   # Statistics
    │   │   └── test_integration.py             # End-to-end pipeline
    │   └── ushanka/
    │       └── test_ushanka.py                 # Projection, merge, init, end-to-end LEP (17 tests)
│
└── data/                                   # Sample data for tests
    ├── sample_train.jsonl
    └── sample_val.jsonl
```

---

## Module Descriptions

### `ruadapt/tokenization/`

Handles all tokenizer manipulation: extending vocabularies with new tokens, replacing tokenizer embeddings, shrinking vocabularies, BPE merge tree operations, and evaluating tokenizer quality.

**Key classes/functions**:
- `core.py`: BPE merge learning (`learn_bpe_fast`), vocabulary injection, tokenizer saving
- `replace.py`: Embedding re-initialization (`reinit_embeddings_with_head_universal`) — modes: random, mean, weighted_mean
- `merges.py`: Fast BPE merge learning in tiktoken format (heap-based)
- `convert.py`: Format converters (tiktoken ↔ HF, vocab extraction, freq list)
- `bpe_tree.py`: `build_merge_tree()`, `recursive_split()` — shared with `training.datasets`
- `utils.py`: `get_tokenizer_properties()`, `convert_token_universal()`, `get_first_diff_id()`, `get_special_token_ids()`, `get_new_token_ids()`, `get_filler_ids()`, `get_trainable_ids()`
- `trim.py`: `classify_tokens()`, `cascade_remove()`, `rebuild_tokenizer()`, `trim_tokenizer()` — freq-aware vocab pruning
- `trim_model.py`: `build_new_embeddings()`, `trim_model()` — resize model after trimming
- `cli.py`: CLI entrypoint for tokenizer replacement

### `ruadapt/initialization/`

Research pipeline for initializing embeddings when extending a vocabulary. Decomposes new tokens into subwords, passes them through a frozen LLM, and trains an MLP head to predict optimal embeddings.

**Pipeline**: Data Prep → Cache Generation → Head Training → Evaluation

**Key classes/functions**:
- `data/build_bpe_dataset.py`: Builds fragmentation dataset (train/val split at token level)
- `data/build_smart_dataset.py`: K-coverage dataset (every new token appears K+ times)
- `data/streamer.py`: `FilteredDatasetStreamer` — HF streaming wrapper with filtering
- `cache/precompute.py`: Precomputes hidden states from all 25 layers
- `head/train.py`: `LayerAttentionHead` architecture + training from precomputed cache
- `head/metrics.py`: `FastMetrics` — loss functions (mse, cosine, cosine_norm, etc.) and eval metrics (logit_mrr, centered_cos_dist, norm_error)
- `eval/micro_cpt.py`: Mini-CPT golden metric (1000 steps, measures Calibrated PPL)
- `eval/delta_w.py`: Delta-W diagnostic — decomposes weight change into radial + tangential

### `ruadapt/training/`

Unified training core for CPT, CLM, and SFT. Config-driven with factory pattern for datasets. From dgx_llm.

**Key classes/functions**:
- `config/schema.py`: `MainConfig`, `ModelConfig`, `DataConfig`, `LoRAConfig`, `FreezeConfig`, `TrainingConfig`
- `core/model.py`: `load_model_and_tokenizer()`, `apply_lora_with_tied_embeddings()` — tied weights workaround for Qwen3.5
- `core/trainer.py`: `UnifiedTrainer(Trainer)` — FSDP checkpoint saving, differential weight decay, WSD scheduler
- `core/distributed.py`: `init_distributed()`, `wrap_model()` — FSDP/DDP/DeepSpeed
- `core/freeze.py`: `freeze_all_except()`, `register_embed_freeze_hook()`
- `datasets/factory.py`: `DatasetFactory(Protocol)`, `CollatorFactory(Protocol)`, `load_factory(dotpath)`
- `datasets/unified_factory.py`: `UnifiedDatasetFactory` — CPT/CLM/Substitution
- `datasets/sft_factory.py`: `SFTDatasetFactory` — chat-template SFT
- `datasets/packing.py`: `PackedDataset`, BPE-Dropout substitution (random + targeted)

### `ruadapt/ushanka/`

LEP (Layer Embedding Projection) — composes an adapted base model into an instruct version.

**Key classes/functions**:
- `compose.py`: LEP orchestrator — loads base, instruct, and donor models, calls `make_ushanka()`
- `merge.py`: `make_ushanka()` — model merging via embedding projection
- `projection.py`: `PROJECTION_MODES` dict — straight, union, conversion, bypass, sum_conversion, occ_conversion

### `ruadapt/evaluation/`

Model evaluation via llmtf_open (Russian LLM benchmarks: darumeru, MMLU-ru, rucola, etc.).

**Key components**:
- `llmtf_open/` — Git submodule from RefalMachine/llmtf_open
- `runner.py` — Wrapper for running evaluations from the pipeline

### `ruadapt/inference/`

Batch inference using vLLM.

**Key components**:
- `vllm.py`: `infer_vllm()` — batch generation with vLLM
- `utils.py`: Re-exports from `ruadapt.utils.io`

### `ruadapt/utils/`

Shared utilities used across modules.

**Key components**:
- `io.py`: `read_jsonl()`, `write_jsonl()`, `read_json()`
- `seed.py`: `set_random_seed()`
- `model_utils.py`: `load_causal_lm()` — Qwen3.5-aware model loader (consolidated, bug-free)

### `deprecated/`

Old code that has not been migrated to the new architecture. Contains instruct tuning (DPO/KTO/CPO/SMPO with Unsloth), old HF Trainer pretraining, old pipeline orchestrators, and old pipeline configs. For reference only — do not import from here in new code.

---

## Data Flow

```
                    ┌─────────────────────────┐
                    │   Data Preparation       │
                    │   (initialization/data)  │
                    └──────────┬──────────────┘
                               │
                    ┌──────────▼──────────────┐
                    │   Cache Generation       │
                    │   (initialization/cache) │
                    └──────────┬──────────────┘
                               │
                    ┌──────────▼──────────────┐
                    │   Head Training          │
                    │   (initialization/head)  │
                    └──────────┬──────────────┘
                               │
┌──────────────┐    ┌──────────▼──────────────┐
│ Tokenization │───▶│   Embedding Injection    │
│ (tokenization)│   │   (tokenization/replace) │
└──────────────┘    └──────────┬──────────────┘
                               │
                    ┌──────────▼──────────────┐
                    │   CPT                    │
                    │   (training/train)       │
                    └──────────┬──────────────┘
                               │
                    ┌──────────▼──────────────┐
                    │   LEP / Ushanka          │
                    │   (ushanka/compose)      │
                    └──────────┬──────────────┘
                               │
                    ┌──────────▼──────────────┐
                    │   SFT + SimPO/DPO        │
                    │   (training/train)       │
                    └──────────┬──────────────┘
                               │
                    ┌──────────▼──────────────┐
                    │   Evaluation             │
                    │   (evaluation/runner)    │
                    └─────────────────────────┘
```

---

## CLI Entrypoints

Each major pipeline step is a separate invocation:

```bash
# Tokenizer extension
python -m ruadapt.tokenization.core --config configs/tokenizer/qwen_extend.json

# Tokenizer replacement (embedding init)
python -m ruadapt.tokenization.cli --model_path MODEL --tokenizer_path TOK --mode mean

# Tokenizer trimming (freq-aware pruning)
python -m ruadapt.tokenization.trim --model_path MODEL --data_path DATA --output_dir OUT

# Model resize after trimming
python -m ruadapt.tokenization.trim_model --model_path MODEL --trim_dir TRIM --output_dir OUT

# Head training
python -m ruadapt.initialization.head.train --cache_dir cache/ --output head.pt

# Continued Pretraining
python -m ruadapt.training.train --config configs/cpt/u128_cpt.json

# LEP composition
python -m ruadapt.ushanka.compose --config ushanka/configs/qwen35_lep.json

# SFT
python -m ruadapt.training.train --config configs/sft/sft_default.json

# Evaluation
python -m ruadapt.evaluation.runner --model_path ./adapted_model --tasks darumeru,mmlu_ru

# Batch inference
python -m ruadapt.inference.vllm MODEL_PATH INPUT_PATH OUTPUT_PATH
```

Complex multi-step pipelines are composed via shell scripts in `scripts/`.
