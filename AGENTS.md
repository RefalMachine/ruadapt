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

## Project State (2026-08)

**Migration complete (2026-06); current focus is production SFT of Qwen3.5-27B.**

Code consolidated from three sources into a single `ruadapt` package:

| Source | What it provided | Status |
|--------|-----------------|--------|
| `devel/ruadapt` (this repo) | Tokenization, Ushanka/LEP, evaluation, inference | Restructured in-place |
| `dgx_llm` | Training core (CPT/CLM/SFT), config-driven, factory pattern | Migrated into `ruadapt/training/` |
| `tokenizer_init_research` | Data prep, head initialization, analytics | Migrated into `ruadapt/initialization/` |

Old code moved to `deprecated/` for reference. llmtf_open submodule updated to latest main with local Qwen3.5/vLLM fixes.

### SFT throughput milestones (Qwen3.5-27B, LoRA r=128, H100 NVL 94GB)

| Version | Regime | non-padding tok/s | Note |
|---|---|---|---|
| v7 | bf16, dense, grad ckpt, micro-batch ~400 tok | 2219 (4 GPU) | baseline, memory-bound |
| v8 | bf16, **packing** chunk 4096, grad ckpt | 3695 (2 GPU) | `configs/sft_v8*.json` |
| v9 | **fp8 storage** of frozen base, chunk 3072, **no grad ckpt** | 4418 (2 GPU) | `configs/sft_v9_fp8_2gpu.json` |
| **v9** | same, 4 GPU | **8753** | full run: 4908 steps / 7.3 h, `configs/sft_v9_fp8_4gpu.json` |

Total ×3.9 vs v7 on the same hardware. The regime is now compute-bound
(GEMM ≈60% of step time at ~500 TFLOPS). Details: [PERF_PLAN.md](PERF_PLAN.md).

---

## Design Principles

- **Modularity**: each module (tokenization, initialization, training, ushanka, evaluation) is self-contained with clear interfaces
- **Config-driven**: training uses JSON configs with HfArgumentParser dataclasses (from dgx_llm)
- **Factory pattern**: dataset creation is pluggable via `DatasetFactory` + `CollatorFactory` protocols
- **Composable freeze/LoRA**: freeze strategies and LoRA are composable modules, not hardcoded
- **No Unsloth dependency**: training uses native Transformers + PEFT + FSDP
- **CLI per step**: each pipeline stage is a separate `python -m ruadapt.*` invocation, composed via shell scripts
- **Subword decomposition**: new tokens are initialized by decomposing them into existing subwords and predicting embeddings via a trained MLP head or mathematical baseline (mean/wmean)
- **Tied embeddings** (2B research path): `input_embeddings` and `lm_head` share one weight matrix — predicted vectors must work in both roles. The 27B production model has `tie_word_embeddings=False`

---

## Module Overview

| Module | Purpose | Key entry points |
|--------|---------|-----------------|
| `ruadapt/tokenization/` | Tokenizer extension, replacement, shrinking, BPE tree, **trim** | `core.py`, `replace.py`, `merges.py`, `bpe_tree.py`, `trim.py`, `trim_model.py`, `utils.py` |
| `ruadapt/initialization/` | Embedding initialization research | `data/build_bpe_dataset.py`, `head/train.py`, `eval/micro_cpt.py` |
| `ruadapt/training/` | Unified training core (CPT/CLM/SFT), SFT packing, fp8 storage | `train.py` (CLI entrypoint), `core/fp8.py`, `datasets/sft_factory.py` |
| `ruadapt/ushanka/` | LEP — Layer Embedding Projection | `compose.py`, `merge.py`, `projection.py` |
| `ruadapt/evaluation/` | llmtf_open submodule + runner | `runner.py` |
| `ruadapt/inference/` | vLLM batch inference | `vllm.py` |
| `ruadapt/utils/` | Shared utilities + text_only LoRA adapter fix | `io.py`, `seed.py`, `model_utils.py`, `adapter.py` (CLI: `python -m ruadapt.utils.adapter`) |
| `deprecated/` | Old code not yet migrated to new architecture | `instruct_tuning/`, `pretraining/`, `root_scripts/`, `pipeline_configs/` |

See [STRUCTURE.md](STRUCTURE.md) for full directory tree and file descriptions.

---

## Testing Rules

- **Framework**: pytest
- **Command**: `pytest tests/ -v`  (perf benchmarks live in `tests/perf/` and are
  **not** collected by pytest)
- **Current coverage** (140 tests: 28 tokenization, 102 training, 10 utils):
  - `tests/tokenization/` — BPE tree, format converters, token utils
  - `tests/training/` — config, datasets, collators, factories, SFT packing,
    fp8 storage (23 tests), integration
  - `tests/utils/` — adapter key fixing (text_only → multimodal)
  - `tests/initialization/`, `tests/ushanka/` — **empty (only `__init__.py`)**.
    Old docs claimed 56 initialization + 17 ushanka tests; those files are not
    on disk. Restore before relying on them — do not assume the modules are tested.
- **Known env dependency**: several `tests/training/` tests point at the local
  model `/workdir/models/RuadaptQwen3.5-2B-Base-u128_*`. If `/workdir/models/`
  is absent they ERROR (OSError about repo id), which is an environment gap,
  not a code bug. Self-contained subset: fp8 storage, SFT packing, adapter,
  tokenization.
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

**Production SFT of Qwen3.5-27B.** Migration (Stages 0–6) and the fp8 training
path are done; the v9 full run is finished and merged models exist.

Open items:

1. **Training schedule decision** — the v9 full run overfits after ~1 epoch
   (eval_loss min 0.5171 @ step 1500 → 0.5876 @ 4908) and the best checkpoint
   was not saved (`save_total_limit=2`). Configs deliberately left untouched;
   see [PERF_PLAN.md](PERF_PLAN.md) §3.1
2. **Speed** — apply `ddp_find_unused_parameters: false` (+1–3%); the only large
   reserve is W8A8 fp8-compute (+25–33%), blocked by cublasLt < 12.9
   ([PERF_PLAN.md](PERF_PLAN.md) §3.2–3.3)
3. **Quality gate** — parity check of the merge path (adapter over fp8 base vs
   over bf16 base), [PERF_PLAN.md](PERF_PLAN.md) §3.5
4. **End-to-end pipeline validation** on the 2B research path
   ([PIPELINE_TEST_PLAN.md](PIPELINE_TEST_PLAN.md), not run yet)
5. Preference optimization: port SMPO / extend the training core with
   DPO/KTO/CPO when needed
6. Restore missing test suites: `tests/initialization/` and `tests/ushanka/`
   are empty on disk although migration docs claimed them complete

Closed: package migration (2026-06), SFT packing (v8), fp8 storage + full run
(v9).

---

## Key Technical Details

### Current production model — Qwen3.5-27B (SFT, LoRA)

- **Base weights**: `Qwen/Qwen3.5-27B` (bf16, 55.6 GB) or
  `Qwen/Qwen3.5-27B-FP8` (28.8 GiB, vendor E4M3 block-128 quantization)
- **Architecture**: `Qwen3_5ForConditionalGeneration` (multimodal wrapper);
  training uses `text_only: true` → `Qwen3_5ForCausalLM` (−0.86 GiB, no vision)
- **Text tower**: 64 layers — **48 GatedDeltaNet** (linear attention, FLA
  kernels) + **16 full-attention**; hidden 5120, FFN 17408, vocab 248 320,
  **`tie_word_embeddings=False`** (`embed_tokens` and `lm_head` are separate)
- **LoRA**: r=128, alpha=128, 7 targets (q/k/v/o/gate/up/down), 637.5M
  trainable = 2.32%, adapters in fp32
- **SFT packing**: greedy bin-packing of whole samples into chunks;
  requires `per_device_train_batch_size=1` (FLA varlen) and
  `remove_unused_columns=false` (else `seq_idx`/`cu_seq_lens_q` are dropped and
  document isolation silently breaks)
- **fp8 storage**: `fp8_storage: true` + `fp8_compile_dequant: true` keep the
  frozen base in fp8 and dequantize inside a custom `autograd.Function`; this
  frees ~30 GiB/GPU and lets `gradient_checkpointing: false`. See
  [FP8.md](FP8.md)
- **Stack**: torch 2.11.0+cu128, transformers 5.9.0, peft 0.19.1,
  liger_kernel 0.8.1, flash_attn_3 3.0.0, flash-linear-attention 0.4.0,
  vllm 0.21.0; 8× H100 NVL 94GB
- **Multi-GPU**: `torchrun --nproc_per_node=N` with DDP (v8/v9 run on 2–4 GPUs);
  use explicit `--master_port` when another job is already running
- **Merge**: adapters trained over the FP8 base are merged into the **bf16**
  base by rewriting `base_model_name_or_path` in `adapter_config.json`
  ([FP8.md](FP8.md) §7.2) — quality of this shortcut is not yet measured

### Research path — Qwen3.5-2B (tokenization / embedding initialization)

- **Target model**: Qwen3.5-2B-Base (Ruadapt variant, ~291K vocab, ~248K frozen + ~43K new)
- **Tied embeddings** (2B only): `embed_tokens` and `lm_head` share weights — special handling needed for LoRA
- **Embedding dim**: 2048, 25 transformer layers
- **BPE tree**: shared between `tokenization/bpe_tree.py` and `training/datasets/` (imported from tokenization)
- **text_only LoRA keys**: adapters trained with `text_only=true` (`Qwen3_5ForCausalLM`) save keys without the `language_model.` segment, so they fail to load into the multimodal `Qwen3_5ForConditionalGeneration` during merge. Fix with `python -m ruadapt.utils.adapter --input_dir <adapter> --verify` before `scripts/merge_lora.py`.

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
| SFT packing | Bin-packing several dialogs into one fixed-length chunk with full document isolation (`position_ids` reset + `seq_idx` + `cu_seq_lens_q`) |
| fill | Share of useful (non-padding) tokens in a packed chunk: 0.9246 @2048, 0.9491 @3072, 0.9599 @4096 |
| fp8 storage | Frozen base kept in fp8, compute in bf16 (QLoRA-style): dequant inside a custom `autograd.Function` |
| W8A8 | Weights **and** activations in fp8 — real fp8 compute; the remaining speed reserve, blocked by cublasLt < 12.9 |
| GatedDeltaNet | Linear-attention layer (FLA kernels); 48 of 64 layers in Qwen3.5-27B, the main consumer of activation memory |
| v7 / v8 / v9 | SFT generations: dense → packing → packing + fp8 storage |

---

## References

- [PERF_PLAN.md](PERF_PLAN.md) — **active speed plan** + hardware/memory/fill reference facts
- [FP8.md](FP8.md) — fp8 storage: design, implementation, merge, how to run
- [FP8_REVIEW.md](FP8_REVIEW.md) — 2026-08-14 review: measurements behind the plan
- [STRUCTURE.md](STRUCTURE.md) — Full project directory tree
- [PIPELINE_TEST_PLAN.md](PIPELINE_TEST_PLAN.md) — end-to-end validation plan for the 2B research path (open)
- [MIGRATION_AND_REFACTORING_PLAN.md](MIGRATION_AND_REFACTORING_PLAN.md) — archive: completed migration
- `trash/ANALYSYS.md`, `trash/FP8_CHECK.md` — archive: v7→v8 investigation log, closed fp8 audit
- [README.md](README.md) — Project overview
- Paper: https://arxiv.org/pdf/2312.02598.pdf
