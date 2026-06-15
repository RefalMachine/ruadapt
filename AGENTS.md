# AGENTS.md — RuAdapt

## What This Project Is

RuAdapt is a toolkit for **adapting English-centric LLMs to Russian language**. The core idea: extend the tokenizer with Russian-specific tokens, initialize their embeddings smartly, run continued pretraining (CPT), then compose the adapted base model into an instruct version via LEP (Layer Embedding Projection) and run SFT + preference optimization (SimPO/DPO).

### Product Pipeline

```
Base Model
  → Extend Tokenizer (add Russian tokens)
  → Initialize Embeddings (mean / weighted_mean / trained MLP head)
  → CPT (Continued Pretraining on Russian corpus)
  → Adapted Base Model
  → LEP (Layer Embedding Projection into instruct version)
  → SFT + SimPO/DPO
  → Adapted Instruct Model
```

Each stage is a separate CLI entrypoint. Complex multi-step pipelines are composed via shell scripts in `scripts/`.

### Key Paper

Tikhomirov, Chernyshev. "Impact of Tokenization on LLaMa Russian Adaptation" (arXiv:2312.02598)

---

## Project State (2026-06)

**Migration complete.** Code consolidated from three sources into a single `ruadapt` package:

| Source | What it provided | Status |
|--------|-----------------|--------|
| `devel/ruadapt` (this repo) | Tokenization, Ushanka/LEP, evaluation, inference | Restructured in-place |
| `dgx_llm` | Training core (CPT/CLM/SFT), config-driven, factory pattern | Migrated into `ruadapt/training/` |
| `tokenizer_init_research` | Data prep, head initialization, analytics | Migrated into `ruadapt/initialization/` |

Old code moved to `deprecated/` for reference. llmtf_open submodule updated to latest main with local Qwen3.5/vLLM fixes.

---

## Design Principles

- **Modularity**: each module (tokenization, initialization, training, ushanka, evaluation) is self-contained with clear interfaces
- **Config-driven**: training uses JSON configs with HfArgumentParser dataclasses (from dgx_llm)
- **Factory pattern**: dataset creation is pluggable via `DatasetFactory` + `CollatorFactory` protocols
- **Composable freeze/LoRA**: freeze strategies and LoRA are composable modules, not hardcoded
- **No Unsloth dependency**: training uses native Transformers + PEFT + FSDP
- **CLI per step**: each pipeline stage is a separate `python -m ruadapt.*` invocation, composed via shell scripts
- **Subword decomposition**: new tokens are initialized by decomposing them into existing subwords and predicting embeddings via a trained MLP head or mathematical baseline (mean/wmean)
- **Tied embeddings**: `input_embeddings` and `lm_head` share one weight matrix — predicted vectors must work in both roles

---

## Module Overview

| Module | Purpose | Key entry points |
|--------|---------|-----------------|
| `ruadapt/tokenization/` | Tokenizer extension, replacement, shrinking, BPE tree, **trim** | `core.py`, `replace.py`, `merges.py`, `bpe_tree.py`, `trim.py`, `trim_model.py`, `utils.py` |
| `ruadapt/initialization/` | Embedding initialization research | `data/build_bpe_dataset.py`, `head/train.py`, `eval/micro_cpt.py` |
| `ruadapt/training/` | Unified training core (CPT/CLM/SFT) | `train.py` (CLI entrypoint) |
| `ruadapt/ushanka/` | LEP — Layer Embedding Projection | `compose.py`, `merge.py`, `projection.py` |
| `ruadapt/evaluation/` | llmtf_open submodule + runner | `runner.py` |
| `ruadapt/inference/` | vLLM batch inference | `vllm.py` |
| `ruadapt/utils/` | Shared utilities | `io.py`, `seed.py`, `model_utils.py` |
| `deprecated/` | Old code not yet migrated to new architecture | `instruct_tuning/`, `pretraining/`, `root_scripts/`, `pipeline_configs/` |

See [STRUCTURE.md](STRUCTURE.md) for full directory tree and file descriptions.

---

## Testing Rules

- **Framework**: pytest
- **Command**: `pytest tests/ -v`
- **Structure**: each module has a corresponding test directory under `tests/`
  - `tests/tokenization/` — tokenizer operations
  - `tests/initialization/` — BPE tree, data builders, head training, metrics
  - `tests/training/` — config, datasets, collators, factories, integration
  - `tests/ushanka/` — LEP composition
- **Before committing**: run existing tests for any module you changed
- **New logic**: write tests with the PR. Tests should be self-contained (use fixtures in `conftest.py`, avoid external model paths where possible)
- **Test data**: use small synthetic fixtures, not production data paths

---

## Working Rules

1. **Read before write**: always scan existing code and utilities before writing new ones
2. **No `rm`**: never delete files. Move to `trash/` directory if cleanup is needed
3. **Project boundary**: do not delete or modify files outside the project directory (`devel/ruadapt/`). Exception: working with models in `/workdir/models/` (loading, saving, inspecting)
4. **Imports**: use `from ruadapt.module import ...` — the package is pip-installable
5. **No Unsloth**: the new training core does not use Unsloth. Use Transformers + PEFT + FSDP
6. **Config format**: training configs are JSON, parsed by HfArgumentParser dataclasses. See `ruadapt/training/config/schema.py`
7. **Factory pattern**: new dataset formats implement `DatasetFactory` + `CollatorFactory` protocols from `ruadapt/training/datasets/factory.py`
8. **Distributed**: multi-GPU via torchrun + FSDP (default) or DDP or DeepSpeed
9. **transformers 5.x**: use `processing_class` not `tokenizer` in Trainer

---

## Current Goal

**Continue development.** Migration is complete (Stages 0–6). Next steps:

1. ~~Write remaining tests (initialization, ushanka)~~ — Done (2026-06-04): 56 initialization + 17 ushanka tests
2. Run end-to-end pipeline validation
3. Port SMPO trainer when preference training is needed
4. Extend training core with DPO/KTO/CPO support

---

## Key Technical Details

- **Target model**: Qwen3.5-2B-Base (Ruadapt variant, ~291K vocab, ~248K frozen + ~43K new)
- **Tied embeddings**: `embed_tokens` and `lm_head` share weights — special handling needed for LoRA
- **Architecture**: Qwen3.5 uses `Qwen3_5ForConditionalGeneration` (conditional generation)
- **Embedding dim**: 2048, 25 transformer layers
- **Single-GPU**: `CUDA_VISIBLE_DEVICES=0` required (fla kernel issue)
- **BPE tree**: shared between `tokenization/bpe_tree.py` and `training/datasets/` (imported from tokenization)

---

## Glossary

| Term | Meaning |
|------|---------|
| CPT | Continued Pretraining — training on Russian text corpus with extended tokenizer |
| LEP | Layer Embedding Projection — composing adapted base model into instruct version (ushanka) |
| SFT | Supervised Fine-Tuning on instruction data |
| SimPO | Simple Margin Preference Optimization |
| DPO | Direct Preference Optimization |
| BPE | Byte Pair Encoding — tokenizer algorithm |
| Tied embeddings | `input_embeddings` and `lm_head` share one weight matrix |
| Norm explosion | Predicted vector norms >> gold norms → softmax calibration break |
| Mini-CPT | 1000-step CPT used as evaluation proxy (golden metric) |
| Calibrated PPL | PPL after mini-CPT |
| Delta-W | Weight change during CPT: `post_emb - pre_emb`. Diagnostic signal. |
| K-coverage | Dataset where every new token appears at least K times |
| freeze_idx | Token ID boundary: tokens below this are frozen base vocabulary |

---

## References

- [STRUCTURE.md](STRUCTURE.md) — Full project directory tree
- [MIGRATION_AND_REFACTORING_PLAN.md](MIGRATION_AND_REFACTORING_PLAN.md) — Migration status and plan
- [README.md](README.md) — Project overview
- Paper: https://arxiv.org/pdf/2312.02598.pdf
