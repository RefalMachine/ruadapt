# Migration and Refactoring Plan

> **ARCHIVE (kept for reference, 2026-08-14).** All stages (0–6) are complete;
> every checkbox in this document is closed. The package structure it describes
> is the current one — see [STRUCTURE.md](STRUCTURE.md) for the live tree and
> [AGENTS.md](AGENTS.md) for the current project state and goals. No further
> action is expected from this plan.

## Overview

**Migration complete (2026-05).** Code consolidated from three sources into a single `ruadapt` package — research-first, but well-structured and convenient for all pipeline stages.

### Source Repositories

| Source | Location | What it provides |
|--------|----------|-----------------|
| **devel/ruadapt** (this repo) | `/workdir/devel/ruadapt/` | Tokenization, Ushanka/LEP, evaluation (llmtf_open), inference, instruct tuning |
| **dgx_llm** | `/workdir/dgx_llm/` | Training core (CPT/CLM/SFT), config-driven, factory pattern, tests (~4200 LOC) |
| **tokenizer_init_research** | `/workdir/tokenizer_init_research/` | Data prep, head initialization, analytics, evaluation (~5500 LOC) |

### Pipeline

```
Base Model
  → Extend Tokenizer (add Russian tokens)
  → Initialize Embeddings (mean / wmean / trained MLP head)
  → CPT (Continued Pretraining)
  → Adapted Base Model
  → LEP (Layer Embedding Projection via Ushanka)
  → SFT + SimPO/DPO
  → Adapted Instruct Model
```

Each stage is a separate CLI entrypoint. No unified orchestrator — complex pipelines are composed via shell scripts.

---

## Target Architecture

```
devel/ruadapt/
├── pyproject.toml                    # Unified pip-installable package
├── README.md
├── AGENTS.md
├── STRUCTURE.md
├── MIGRATION_AND_REFACTORING_PLAN.md
│
├── ruadapt/                          # Main package
│   ├── __init__.py                   # Version
│   ├── tokenization/                 # Tokenizer manipulation + BPE tree
│   ├── initialization/               # Embedding init research (data, cache, head, eval)
│   ├── training/                     # Unified training core (CPT/CLM/SFT)
│   ├── ushanka/                      # LEP composition
│   ├── evaluation/                   # llmtf_open submodule + wrapper
│   ├── inference/                    # vLLM batch inference
│   └── utils/                        # Shared utilities (io, seed, model_utils)
│
├── deprecated/                       # Old code (reference only)
│   ├── instruct_tuning/
│   ├── pretraining/
│   ├── root_scripts/                 # Old pipeline orchestrators
│   └── README.md
│
├── configs/                          # Training configs (dgx_llm JSON format)
├── deepspeed_configs/
├── scripts/                          # Shell + Python utility scripts
├── tests/                            # Full test suite
└── data/                             # Sample data for tests
```

---

## Key Architectural Decisions

### 1. BPE tree lives in `tokenization/`

`bpe_tree.py` (45 lines, pure Python, zero deps) is fundamentally a tokenizer operation. Both `training.datasets` and `initialization` import from `ruadapt.tokenization.bpe_tree`.

```python
from ruadapt.tokenization.bpe_tree import build_merge_tree, recursive_split
```

This avoids `training → initialization` dependency and keeps the dependency graph acyclic.

### 2. `initialization/` is a research subpackage

It contains data prep, cache generation, head training, and evaluation — needed for the initialization stage only. Not used at CPT/SFT/inference runtime. Optional heavy deps (matplotlib, scipy) are listed as extras in `pyproject.toml`.

### 3. CLI entrypoints per pipeline step

No unified orchestrator. Each major step is invoked separately:

```bash
# Tokenizer extension
python -m ruadapt.tokenization.extend --config configs/tokenizer/qwen_extend.json

# Embedding initialization
python -m ruadapt.initialization.head.train --cache_dir cache/ --output head.pt

# CPT
python -m ruadapt.training.train --config configs/cpt/u128_cpt.json

# LEP composition
python -m ruadapt.ushanka.compose --config ushanka/configs/qwen35_lep.json

# SFT
python -m ruadapt.training.train --config configs/sft/sft_default.json

# Evaluation
python -m ruadapt.evaluation.runner --model_path ./adapted_model --tasks darumeru,mmlu_ru
```

Complex multi-step pipelines are composed via shell scripts in `scripts/`.

### 4. Old root-level scripts are deprecated

`run_pipeline.py`, `run_pipeline_config.py`, `extend_or_replace.py`, `mix_data.py`, `pipeline_configs/` — all reference old module paths and Unsloth-based code. They move to `deprecated/root_scripts/` for reference. New pipeline scripts go in `scripts/`.

### 5. `_load_causal_lm()` consolidated in `utils/model_utils.py`

Duplicated 6 times in tokenizer_init_research (3 copies have a copy-paste bug where `ModelClass = AutoModelForCausalLM` overwrites the Qwen3.5 class). Single canonical implementation in `ruadapt/utils/model_utils.py`.

---

## Dependency Map

```
ruadapt.utils                    (standalone: io, seed, model_utils)
    ↑
ruadapt.tokenization             (standalone + bpe_tree.py)
    ↑              ↑
ruadapt.initialization     ruadapt.training.datasets
(initializes new tokens)   (imports bpe_tree from tokenization)
                                ↑
                          ruadapt.training.core
                          (model, trainer, freeze, distributed)
                                ↑
                          ruadapt.ushanka
                          (standalone, loads models directly)
                                ↑
                          ruadapt.evaluation  (standalone, llmtf_open)
                          ruadapt.inference   (standalone, vllm)
```

**No cycles.** `tokenization` is the leaf dependency. `initialization` and `training` both depend on `tokenization` but not on each other.

---

## Migration Stages

### Stage 0: Infrastructure Setup

**Goal**: Create package structure, pyproject.toml, move deprecated code.

**Tasks**:
- [x] Create `pyproject.toml` with merged dependencies
- [x] Create `ruadapt/__init__.py` with `__version__`
- [x] Create all `__init__.py` files for subpackages:
  - `ruadapt/tokenization/__init__.py` (already exists, keep)
  - `ruadapt/initialization/__init__.py`
  - `ruadapt/initialization/data/__init__.py`
  - `ruadapt/initialization/cache/__init__.py`
  - `ruadapt/initialization/head/__init__.py`
  - `ruadapt/initialization/eval/__init__.py`
  - `ruadapt/initialization/analytics/__init__.py`
  - `ruadapt/training/__init__.py`
  - `ruadapt/training/config/__init__.py`
  - `ruadapt/training/core/__init__.py`
  - `ruadapt/training/datasets/__init__.py`
  - `ruadapt/ushanka/__init__.py`
  - `ruadapt/evaluation/__init__.py`
  - `ruadapt/inference/__init__.py`
  - `ruadapt/utils/__init__.py`
- [x] Move `ruadapt/instruct_tuning/` → `deprecated/instruct_tuning/`
- [x] Move `ruadapt/pretraining/` → `deprecated/pretraining/`
- [x] Move root scripts (`run_pipeline.py`, `run_pipeline_config.py`, `run_pipeline_app.py`, `run_pipeline_infer.py`, `extend_or_replace.py`, `mix_data.py`) → `deprecated/root_scripts/`
- [x] Move `pipeline_configs/` → `deprecated/pipeline_configs/`
- [x] Update `deprecated/README.md` with full inventory
- [x] Create `tests/conftest.py` with shared fixtures (small tokenizer, sample texts)
- [x] Create `data/` directory with `sample_train.jsonl`, `sample_val.jsonl`
- [x] Update `.gitmodules`: llmtf_open submodule → latest main

**`pyproject.toml` dependencies**:
```toml
[project]
name = "ruadapt"
version = "0.1.0"
requires-python = ">=3.10"

dependencies = [
    # Core (from dgx_llm)
    "torch>=2.1.0",
    "transformers>=4.40.0",
    "peft>=0.10.0",
    "datasets>=2.18.0",
    "accelerate>=0.28.0",
    "deepspeed>=0.14.0",
    # Tokenization
    "tiktoken",
    "sentencepiece",
    "regex",
    "safetensors",
]

[project.optional-dependencies]
eval = ["vllm", "bitsandbytes"]
analytics = ["matplotlib", "numpy", "scipy"]
dev = ["pytest", "ruff"]
all = ["ruadapt[eval,analytics,dev]"]

[project.scripts]
ruadapt-train = "ruadapt.training.train:main"
```

**Verification**: `pip install -e .` succeeds, `import ruadapt` works, `python -c "import ruadapt; print(ruadapt.__version__)"` prints `0.1.0`.

---

### Stage 1: Tokenization Module

**Goal**: Migrate tokenizer manipulation code. Establish `bpe_tree.py` as canonical shared module.

**Source**: `devel/ruadapt/ruadapt/tokenization/` (current files) + `tokenizer_init_research/utils/tokenizer_utils.py`

**File mapping**:

| Source | Target | Notes |
|--------|--------|-------|
| `extend_tokenizer.py` (480 lines) | `tokenization/core.py` | BPE learning, vocab injection. Rename + keep all functions |
| `replace_tokenizer.py` (211 lines) | `tokenization/replace.py` | Embedding reinit (mean/wmean/random) |
| `shrink_tokenizer.py` (36 lines) | `tokenization/shrink.py` | Vocab truncation |
| `add_merges_fast.py` (288 lines) | `tokenization/merges.py` | Fast BPE merge learning (tiktoken, heap-based) |
| `convert_tiktoken.py` (72 lines) | `tokenization/convert.py` | tiktoken → HF format converter |
| `extract_vocab.py` (127 lines) | `tokenization/convert.py` (merge) | Vocab extraction → append to convert.py |
| `expand_tiktoken_save_hf.py` (104 lines) | `tokenization/convert.py` (merge) | tiktoken extend+convert → append to convert.py |
| `convert_hf_tokenizer_vocab_to_freq_list.py` | `tokenization/convert.py` (merge) | Freq list extraction → append to convert.py |
| `utils.py` (398 lines, ~315 commented) | `tokenization/utils.py` | **Clean up**: remove commented-out dead code (lines 86-398). Keep 8 active functions |
| `run_replace_tokenizer.py` (114 lines) | `tokenization/cli.py` | CLI wrapper for replace. Update imports |
| `train_spm.py` (17 lines) | `tokenization/train_spm.py` | Keep as-is (thin SPM wrapper) |
| `evaluation/` | `tokenization/evaluation/` | Keep directory structure as-is |
| `tokenizer_init_research/utils/tokenizer_utils.py` (38 lines) | Merge into `tokenization/utils.py` | `get_first_diff_id()`, `get_special_token_ids()` — add to existing utils |
| `tokenizer_init_research/pipeline/data_prep/bpe_tree.py` (45 lines) | `tokenization/bpe_tree.py` | **Canonical location** for shared BPE tree |

**Dead code — do NOT migrate**:
| File | Reason |
|------|--------|
| `extend_tokenizer_old.py` (90 lines) | Superseded by `extend_tokenizer.py`. Has naive merge approach |
| `add_merges.py` (240 lines) | Slow version of `add_merges_fast.py`. Both are tiktoken-only |
| `ushanka/apc.py` | Empty file (0 lines) |

**Import changes during rename**:
```python
# In cli.py (was run_replace_tokenizer.py)
from .replace import reinit_embeddings_with_head_universal  # was .replace_tokenizer
from .utils import special_encode  # unchanged

# In replace.py (was replace_tokenizer.py)
from .utils import get_tokenizer_properties, convert_token_to_string_universal, convert_token_universal
# unchanged — already uses relative imports
```

**Tests to write** (`tests/tokenization/`):
- `test_core.py` — test `learn_bpe_fast()` on small vocab, test vocab injection roundtrip
- `test_replace.py` — test `reinit_embeddings_with_head_universal()` with mock embeddings
- `test_merges.py` — test `learn_bpe_fast()` tiktoken format output
- `test_utils.py` — test `get_tokenizer_properties()`, `convert_token_universal()`
- `test_bpe_tree.py` — test `build_merge_tree()`, `recursive_split()` (migrated from dgx_llm tests)
- `test_convert.py` — test format converters

**Verification**: `pytest tests/tokenization/ -v` passes. `python -m ruadapt.tokenization.core --help` works.

---

### Stage 2: Training Core (from dgx_llm)

**Goal**: Migrate the unified training library. All `dgx_llm.*` imports become `ruadapt.training.*`.

**Source**: `/workdir/dgx_llm/dgx_llm/`

**File mapping**:

| Source | Target | Lines | Notes |
|--------|--------|-------|-------|
| `train.py` | `training/train.py` | 280 | CLI entrypoint (`fire.Fire`). Change `dgx_llm.*` → `ruadapt.training.*` imports |
| `config/__init__.py` | `training/config/__init__.py` | — | |
| `config/schema.py` | `training/config/schema.py` | 167 | HfArgumentParser dataclasses. Change `dataset_factory` default paths |
| `core/__init__.py` | `training/core/__init__.py` | — | |
| `core/model.py` | `training/core/model.py` | 155 | `load_model_and_tokenizer()`, `apply_lora_with_tied_embeddings()` |
| `core/distributed.py` | `training/core/distributed.py` | 87 | FSDP/DDP/DeepSpeed init |
| `core/trainer.py` | `training/core/trainer.py` | 261 | `UnifiedTrainer`, WSD scheduler, callbacks |
| `core/freeze.py` | `training/core/freeze.py` | 83 | Freeze/unfreeze + embed gradient hooks |
| `datasets/__init__.py` | `training/datasets/__init__.py` | 17 | Re-exports |
| `datasets/factory.py` | `training/datasets/factory.py` | 64 | `DatasetFactory`, `CollatorFactory` protocols |
| `datasets/bpe_tree.py` | **REMOVED** | — | Use `from ruadapt.tokenization.bpe_tree import ...` |
| `datasets/packing.py` | `training/datasets/packing.py` | 619 | Change bpe_tree import path |
| `datasets/unified_factory.py` | `training/datasets/unified_factory.py` | 279 | CPT/CLM/Sub factory |
| `datasets/sft_factory.py` | `training/datasets/sft_factory.py` | 156 | SFT factory (chat template) |
| `datasets/collators.py` | `training/datasets/collators.py` | 42 | |
| `datasets/in_memory.py` | `training/datasets/in_memory.py` | 64 | |
| `datasets/stats.py` | `training/datasets/stats.py` | 322 | |
| `datasets/utils.py` | `training/datasets/utils.py` | 288 | Tokenization helpers, rank-aware caching |
| `datasets/debug.py` | `training/datasets/debug.py` | 90 | print_sample |
| `scripts/trim_tokenizer.py` | `scripts/trim_tokenizer.py` | 626 | Move to top-level scripts/. Update imports |
| `scripts/trim_model.py` | `scripts/trim_model.py` | 258 | Move to top-level scripts/. Update imports |
| `scripts/analyze_targeted_substitution.py` | `scripts/analyze_targeted_substitution.py` | 282 | Move to top-level scripts/. Update imports |

**Import changes** (systematic find-replace):
```python
# Pattern: dgx_llm.X.Y → ruadapt.training.X.Y
from dgx_llm.config.schema import ...       → from ruadapt.training.config.schema import ...
from dgx_llm.core.model import ...          → from ruadapt.training.core.model import ...
from dgx_llm.core.trainer import ...        → from ruadapt.training.core.trainer import ...
from dgx_llm.core.freeze import ...         → from ruadapt.training.core.freeze import ...
from dgx_llm.core.distributed import ...    → from ruadapt.training.core.distributed import ...
from dgx_llm.datasets.factory import ...    → from ruadapt.training.datasets.factory import ...

# bpe_tree — special case: path changed
from dgx_llm.datasets.bpe_tree import ...   → from ruadapt.tokenization.bpe_tree import ...
```

**Config JSON updates**: All `dataset_factory` and `collator_factory` dotpath values:
```json
"dataset_factory": "dgx_llm.datasets.unified_factory.UnifiedDatasetFactory"
→
"dataset_factory": "ruadapt.training.datasets.unified_factory.UnifiedDatasetFactory"
```

**Tests** (migrate from `/workdir/dgx_llm/tests/`):
- `tests/training/conftest.py` — adapt fixtures (update model path if needed)
- `tests/training/test_config.py` — config parsing
- `tests/training/test_utils.py` — dataset utilities
- `tests/training/test_bpe_tree.py` — already in Stage 1, skip here
- `tests/training/test_collators.py` — collators
- `tests/training/test_unified_dataset.py` — packed dataset
- `tests/training/test_factory.py` — factory protocol
- `tests/training/test_stats.py` — statistics
- `tests/training/test_integration.py` — end-to-end

**Verification**: `pytest tests/training/ -v` passes. `python -m ruadapt.training.train --config configs/smoke/u128_cpt.json` runs (needs GPU).

---

### Stage 3: Initialization Pipeline (from tokenizer_init_research)

**Goal**: Migrate data preparation, head training, and evaluation. Eliminate `sys.path` hacks, consolidate `_load_causal_lm()`.

**Source**: `/workdir/tokenizer_init_research/`

**File mapping**:

| Source | Target | Lines | Notes |
|--------|--------|-------|-------|
| `pipeline/data_prep/build_dataset_bpe.py` | `initialization/data/build_bpe_dataset.py` | 99 | Remove sys.path, use `ruadapt.*` imports |
| `pipeline/data_prep/build_smart_local.py` | `initialization/data/build_smart_dataset.py` | 257 | Remove sys.path. Imports `utils.tokenizer_utils` → `ruadapt.tokenization.utils` |
| `pipeline/data_prep/build_smart_dataset.py` | `initialization/data/build_smart_streaming.py` | 204 | Streaming variant. Remove sys.path |
| `pipeline/data_prep/build_seq_local.py` | `initialization/data/build_seq_dataset.py` | 151 | Remove sys.path |
| `pipeline/data_prep/build_token_passport.py` | `initialization/data/build_token_passport.py` | 287 | Remove sys.path. Import bpe_tree from `ruadapt.tokenization.bpe_tree` |
| `pipeline/data_prep/dataset_streamer.py` | `initialization/data/streamer.py` | 118 | Library class. No changes needed (standalone) |
| `pipeline/data_prep/download_cpt_data.py` | `initialization/data/download.py` | 16 | Minimal |
| `pipeline/training/precompute_hidden_states.py` | `initialization/head/precompute.py` | 506 | Remove `_load_causal_lm()` — use `ruadapt.utils.model_utils` |
| `pipeline/training/train_from_cache.py` | `initialization/head/train.py` | 671 | Contains `LayerAttentionHead`. Remove `_load_causal_lm()` |
| `pipeline/training/metrics.py` | `initialization/head/metrics.py` | 533 | Pure library. No changes needed |
| `pipeline/training/eval_per_token.py` | `initialization/eval/per_token.py` | 430 | Remove sys.path + fix `_load_causal_lm` bug |
| `pipeline/training/eval_init_quality.py` | `initialization/eval/init_quality.py` | 254 | Remove `_load_causal_lm()` |
| `pipeline/training/eval_delta_w.py` | `initialization/eval/delta_w.py` | 642 | Remove sys.path + `_load_causal_lm()` |
| `evaluation/evaluate_micro_cpt.py` | `initialization/eval/micro_cpt.py` | 368 | Fix `_load_causal_lm` bug |
| `evaluation/evaluate_ppl.py` | `initialization/eval/ppl.py` | 116 | Fix `_load_causal_lm` bug |
| `evaluation/evaluate_baselines.py` | `initialization/eval/baselines.py` | 154 | Import metrics from `ruadapt.initialization.head.metrics` |
| `evaluation/run_diagnostic.py` | `initialization/eval/diagnostic.py` | 211 | Remove sys.path. Import evaluate_tokenizer from `ruadapt.tokenization.evaluation` |
| `analytics/analyze_train_results.py` | `initialization/analytics/train_results.py` | 145 | Standalone, no changes |
| `analytics/analyze_token_lifecycle.py` | `initialization/analytics/token_lifecycle.py` | 529 | Remove sys.path. Import tokenizer_utils from `ruadapt.tokenization.utils` |
| `analytics/plot_logs.py` | `initialization/analytics/plot_logs.py` | 124 | Standalone |
| `utils/model_utils.py` | `utils/model_utils.py` | 165 | Move to shared `ruadapt/utils/model_utils.py` (consolidation target) |

**NOT migrated** (dead/generated/one-off):
| Source | Reason |
|--------|--------|
| `pipeline/data_prep/build_dataset_unified.py` | Variant of build_smart_local, superseded |
| `pipeline/data_prep/build_seq_size.py` | Variant of build_seq_local |
| `pipeline/data_prep/profile_streamer.py` | Profiling utility, one-off |
| `pipeline/data_prep/test_tail.py` | Test utility, one-off |
| `evaluation/token_lifecycle_matrix.py` | Variant of lifecycle analysis |
| `analytics/analyze_final_results.py` | One-off analysis script |
| `check_format.py`, `count_lines.py`, `save_unmet_tokens.py` | Root-level one-off utilities |
| `data/` (53 files) | Generated datasets — not code, regenerated by pipeline |
| `cache/` (~96 GB) | Compressed tensor caches — regenerated by `precompute` |
| `ruadapt/` (embedded submodule) | No longer needed after migration |

**Critical fix — `_load_causal_lm()` consolidation**:

Current state: 6 copies, 3 with copy-paste bug.

Canonical implementation in `ruadapt/utils/model_utils.py`:
```python
def load_causal_lm(model_path, device_map="auto", torch_dtype="auto"):
    """Load model with correct architecture class (Qwen3.5-aware)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(model_path)
    model_cls = AutoModelForCausalLM
    if hasattr(config, "architectures") and config.architectures:
        if "Qwen3_5ForConditionalGeneration" in config.architectures:
            from transformers import Qwen3_5ForConditionalGeneration
            model_cls = Qwen3_5ForConditionalGeneration
    return model_cls.from_pretrained(model_path, device_map=device_map, torch_dtype=torch_dtype)
```

All 6 files then use `from ruadapt.utils.model_utils import load_causal_lm`.

**Tests to write** (`tests/initialization/`):
- `test_bpe_tree.py` — already in Stage 1
- `test_data_builders.py` — test BPE dataset builder, smart dataset builder on small fixtures
- `test_metrics.py` — test FastMetrics loss functions with synthetic tensors
- `test_streamer.py` — test FilteredDatasetStreamer
- `test_model_utils.py` — test `load_causal_lm`, `_resolve_model_class`

**Verification**: `pytest tests/initialization/ -v` passes. Head training from a small synthetic cache works.

---

### Stage 4: Ushanka (LEP)

**Goal**: Migrate the LEP composition module. Flatten `src/` subdirectory.

**Source**: `devel/ruadapt/ruadapt/ushanka/`

**File mapping**:

| Source | Target | Lines | Notes |
|--------|--------|-------|-------|
| `compose_ushanka.py` (236 lines) | `ushanka/compose.py` | Rename. Keep `__main__` block. Update relative imports |
| `src/ushanka.py` (144 lines) | `ushanka/merge.py` | `make_ushanka()` — model merging logic |
| `src/ushanka_proj_utils.py` (193 lines) | `ushanka/projection.py` | `PROJECTION_MODES` dict, projection algorithms |
| `ushanka_configs/` (30 files) | `ushanka/configs/` | Rename directory |
| `custom_chat_templates/` (1 file) | `ushanka/templates/` | Rename directory |

**Dead code**:
| File | Reason |
|------|--------|
| `apc.py` | Empty file (0 lines) — delete |

**Import changes**:
```python
# In compose.py (was compose_ushanka.py)
from .merge import make_ushanka           # was .src.ushanka
from .projection import list_projection_modes  # was .src.ushanka_proj_utils

# In merge.py (was src/ushanka.py)
from .projection import PROJECTION_MODES  # was .ushanka_proj_utils
```

**Delete**: `ushanka/src/` directory after flattening.

**Tests**: `tests/ushanka/test_compose.py` — test config loading, test `make_ushanka()` on a tiny mock model if feasible.

**Verification**: `python -m ruadapt.ushanka.compose --help` works. Config loading from `ushanka/configs/` works.

---

### Stage 5: Shared Utilities, Evaluation, Inference

**Goal**: Wire up remaining modules. Create shared utilities.

**Tasks**:

**5a. Shared utilities** (`ruadapt/utils/`):
- [x] Create `utils/__init__.py`
- [x] Create `utils/io.py` — `read_jsonl()`, `write_jsonl()`, `read_json()` (from `inference/utils.py`)
- [x] Create `utils/seed.py` — `set_random_seed()` (from `instruct_tuning/utils.py`)
- [x] Create `utils/model_utils.py` — `load_causal_lm()`, `resolve_model_class()`, `ModelInference` (consolidated from tokenizer_init_research, fixes 3 bugs)
- [x] Move `inference/utils.py` → use `ruadapt.utils.io` (keep `inference/utils.py` as re-export shim or delete)

**5b. Evaluation** (`ruadapt/evaluation/`):
- [x] Update llmtf_open submodule to latest main:
  ```bash
  cd ruadapt/evaluation/llmtf_open && git fetch origin && git checkout main
  ```
- [x] Create `evaluation/runner.py` — thin wrapper that calls llmtf_open programmatically

**5c. Inference** (`ruadapt/inference/`):
- [x] Rename `infer_vllm.py` → `inference/vllm.py`
- [x] Update imports: `from .utils import ...` → `from ruadapt.utils.io import ...`

**5d. Scripts** (`scripts/`):
- [x] Migrate from dgx_llm: `trim_tokenizer.py`, `trim_model.py`, `analyze_targeted_substitution.py`
- [ ] Migrate from tokenizer_init_research: key shell scripts (`run_full_evaluation.sh`, `run_delta_w_analysis.sh`, `run_lifecycle_analysis.sh`)
- [ ] Create new `scripts/run_cpt.sh` — example CPT invocation
- [ ] Create new `scripts/run_sft.sh` — example SFT invocation
- [x] Migrate `fix_config.py`, `fix_configs.sh` for VLM config patching

**Verification**: `python -m ruadapt.evaluation.runner --help` works. `python -m ruadapt.inference.vllm --help` works.

---

### Stage 6: Cleanup and Documentation

**Goal**: Remove old code, update documentation, final verification.

**Tasks**:
- [x] Delete `ruadapt/instruct_tuning/` (now in `deprecated/`)
- [x] Delete `ruadapt/pretraining/` (now in `deprecated/`)
- [x] Delete `ushanka/src/` (flattened into `ushanka/`)
- [x] Delete `ushanka/apc.py` (empty)
- [x] Delete `tokenization/extend_tokenizer_old.py` (dead)
- [x] Delete `tokenization/add_merges.py` (slow version, superseded)
- [x] Delete commented-out code in `tokenization/utils.py` (lines 86-398)
- [x] Update `README.md` with new structure, usage examples
- [x] Update `AGENTS.md` with final structure
- [x] Update `STRUCTURE.md`
- [x] Run full test suite: `pytest tests/ -v`
- [x] Verify `pip install -e .` works cleanly
- [ ] Verify `pip install -e ".[all]"` works (optional deps)

---

## What Gets Deprecated

| Module | Location after migration | Reason | Replacement |
|--------|------------------------|--------|-------------|
| `instruct_tuning/train_sft.py` | `deprecated/instruct_tuning/` | Uses Unsloth | `ruadapt.training` + SFT factory |
| `instruct_tuning/train_dpo.py` | `deprecated/instruct_tuning/` | Uses Unsloth + trl | Future: extend training core |
| `instruct_tuning/train_kto.py` | `deprecated/instruct_tuning/` | Uses Unsloth + trl | Future: extend training core |
| `instruct_tuning/train_cpo.py` | `deprecated/instruct_tuning/` | Uses Unsloth + trl | Future: extend training core |
| `instruct_tuning/train_smpo.py` | `deprecated/instruct_tuning/` | Uses Unsloth | Future: extend training core |
| `instruct_tuning/smpo_trainer.py` | `deprecated/instruct_tuning/` | **Valuable** — port to `ruadapt/training/trainers/smpo.py` when needed |
| `instruct_tuning/train_sft_transformers.py` | `deprecated/instruct_tuning/` | Reference for SFT factory improvements |
| `instruct_tuning/train_sft_fsdp.py` | `deprecated/instruct_tuning/` | Reference for FSDP patterns |
| `instruct_tuning/train_cpo_transformers.py` | `deprecated/instruct_tuning/` | Reference for future CPO |
| `instruct_tuning/train_kto_transformers.py` | `deprecated/instruct_tuning/` | Reference for future KTO |
| `instruct_tuning/train_smpo_transformers.py` | `deprecated/instruct_tuning/` | Reference for future SMPO |
| `instruct_tuning/train_cpo_fsdp.py` | `deprecated/instruct_tuning/` | Reference for FSDP patterns |
| `instruct_tuning/train_unsloth.py` | `deprecated/instruct_tuning/` | Dead code |
| `instruct_tuning/dataset.py` | `deprecated/instruct_tuning/` | Reference for SFT data format |
| `instruct_tuning/dpo_dataset.py` | `deprecated/instruct_tuning/` | Reference for future DPO factory |
| `instruct_tuning/utils.py` | `deprecated/instruct_tuning/` | `set_random_seed()` → `ruadapt/utils/seed.py` |
| `instruct_tuning/datasets/` | `deprecated/instruct_tuning/` | Dataset composition scripts |
| `instruct_tuning/models_configs/` | `deprecated/instruct_tuning/` | Per-model training configs |
| `pretraining/train_trainer.py` | `deprecated/pretraining/` | Replaced by `ruadapt.training.train` |
| `pretraining/utils.py` | `deprecated/pretraining/` | Some utils reusable |
| Root scripts | `deprecated/root_scripts/` | Old pipeline orchestrators |
| `pipeline_configs/` | `deprecated/pipeline_configs/` | Old pipeline configs |

**Future porting note**: `smpo_trainer.py` (1047 lines) implements `SimpleMarginPOTrainer` — custom preference optimization with multiple loss types, winsorization, SFT blending. When preference training is needed, port to `ruadapt/training/trainers/smpo.py` following the `UnifiedTrainer` pattern.

---

## Bugs to Fix During Migration

### 1. ~~`_load_causal_lm()` copy-paste bug (3 files)~~ FIXED

Consolidated into `ruadapt/utils/model_utils.load_causal_lm()`. All 6 files now use the single canonical implementation.

### 2. `sft_factory.py` hardcodes Qwen assistant boundaries

`training/datasets/sft_factory.py` hardcodes `<|im_start|>assistant\n\n\n`. Should be parameterized or auto-detected from tokenizer's chat template.

### 3. ~~`utils.py` contains 315 lines of dead code~~ FIXED

Removed during Stage 1 cleanup. Active functions preserved, `get_first_diff_id()` and `get_special_token_ids()` merged from tokenizer_init_research.

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Import breakage during file renames | High | Stage 1 renames + tests before proceeding |
| bpe_tree import from wrong location | Medium | Explicit import in Stage 1 tests |
| `_load_causal_lm` bug propagation | High | Fix in Stage 3, test `resolve_model_class()` |
| SFT factory hardcoded boundaries | Medium | Parameterize in Stage 2 |
| llmtf_open submodule update breaks eval | Low | Pin to known-good commit; test old benchmarks |
| GPU-only tests can't run in CI | Medium | Separate GPU tests; mark with `@pytest.mark.gpu` |
| transformers 5.x `processing_class` | Low | Already handled in dgx_llm trainer |

---

## Timeline

| Stage | Estimated effort | Dependencies | Parallelizable |
|-------|-----------------|--------------|----------------|
| Stage 0: Infrastructure | 1-2h | None | — |
| Stage 1: Tokenization | 3-4h | Stage 0 | With Stages 2, 4 |
| Stage 2: Training core | 4-6h | Stage 0 | With Stages 1, 4 |
| Stage 3: Initialization | 4-6h | Stage 0, Stage 1 (bpe_tree) | After Stage 1 |
| Stage 4: Ushanka | 1-2h | Stage 0 | With Stages 1, 2 |
| Stage 5: Utils + Eval + Scripts | 2-3h | Stage 1-4 | After all |
| Stage 6: Cleanup | 1-2h | Stage 1-5 | After all |

**Total**: ~16-25 hours (2-3 working days).

Stages 1, 2, 4 can be done in parallel after Stage 0. Stage 3 waits for Stage 1 (needs bpe_tree).

---

## Verification Checklist (Final)

After all stages complete:

```bash
# Package install
pip install -e ".[all]"

# Import check
python -c "from ruadapt.tokenization.core import learn_bpe_fast"
python -c "from ruadapt.tokenization.bpe_tree import build_merge_tree"
python -c "from ruadapt.training.config.schema import MainConfig"
python -c "from ruadapt.training.core.model import load_model_and_tokenizer"
python -c "from ruadapt.training.datasets.factory import DatasetFactory"
python -c "from ruadapt.ushanka.merge import make_ushanka"
python -c "from ruadapt.utils.model_utils import load_causal_lm"
python -c "from ruadapt.utils.io import read_jsonl"

# CLI entrypoints
python -m ruadapt.tokenization.core --help
python -m ruadapt.training.train --help
python -m ruadapt.ushanka.compose --help
python -m ruadapt.evaluation.runner --help

# Tests
pytest tests/ -v

# No old imports remain
grep -r "from dgx_llm" ruadapt/ && echo "FAIL: old dgx_llm imports found"
grep -r "from ruadapt.instruct_tuning" ruadapt/ && echo "FAIL: old instruct_tuning imports found"
grep -r "sys.path" ruadapt/ && echo "FAIL: sys.path hacks found"
```
