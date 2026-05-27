# RuAdapt

## **!! ATTENTION !! REPOSITORY UPDATE IN PROGRESS SINCE 27.05.26**

Toolkit for adapting English-centric LLMs to Russian language.

## Overview

RuAdapt extends an LLM's tokenizer with Russian-specific tokens, initializes their embeddings, runs continued pretraining (CPT), and composes the adapted model into an instruct version via LEP (Layer Embedding Projection).

```
Base Model → Extend Tokenizer → Initialize Embeddings → CPT → LEP → SFT + SimPO → Instruct Model
```

## Installation

```bash
pip install -e .
```

## Package Structure

| Module | Purpose |
|--------|---------|
| `ruadapt/tokenization/` | Tokenizer extension, replacement, shrinking |
| `ruadapt/initialization/` | Embedding initialization (BPE decomposition, MLP head training) |
| `ruadapt/training/` | Unified training core (CPT, CLM, SFT) |
| `ruadapt/ushanka/` | LEP — Layer Embedding Projection |
| `ruadapt/evaluation/` | llmtf_open benchmarks (darumeru, MMLU-ru, etc.) |
| `ruadapt/inference/` | vLLM batch inference |

## Quick Start

### Extend a tokenizer

```bash
python -m ruadapt.tokenization.core \
    --base_tokenizer /path/to/base \
    --new_vocab /path/to/vocab.txt \
    --output_dir /path/to/extended
```

### Initialize embeddings (mean baseline)

```bash
python -m ruadapt.tokenization.replace \
    --model_name_or_path /path/to/base_model \
    --new_tokenizer_path /path/to/extended_tokenizer \
    --output_path /path/to/output \
    --mode mean
```

### Train MLP head for embedding initialization

```bash
# 1. Precompute hidden states (one-time, ~2-3h)
python -m ruadapt.initialization.cache.precompute \
    --cache-dir cache --model-path /path/to/base_model

# 2. Train head (~30s/epoch)
python -m ruadapt.initialization.head.train \
    --pooling attention --loss-type cosine_norm \
    --cache-dir cache --results-dir results/my_run
```

### Run CPT

```bash
python -m ruadapt.training.train --config configs/cpt/u128_cpt.json
```

### Run LEP (compose base into instruct)

```bash
python -m ruadapt.ushanka.compose \
    --config ushanka/configs/qwen3.5_3b.json
```

### Fix adapted model config for VLM evaluation (Qwen3.5)

After CPT, the adapted model's `config.json` loses multimodal fields needed by vLLM:

```bash
python scripts/fix_config.py \
    --original /path/to/Qwen3.5-2B-Base \
    --adapted /path/to/adapted_model
```

## Documentation

- [AGENTS.md](AGENTS.md) — Working rules, project state, design principles
- [STRUCTURE.md](STRUCTURE.md) — Full directory tree with file descriptions
- [MIGRATION_AND_REFACTORING_PLAN.md](MIGRATION_AND_REFACTORING_PLAN.md) — Migration status

## Paper

Tikhomirov, Chernyshev. "Impact of Tokenization on LLaMa Russian Adaptation"
Proceedings of Ivannikov ISPRAS Open Conference (2023)
[arXiv:2312.02598](https://arxiv.org/pdf/2312.02598.pdf)

## Credits

- Instruction tuning code: based on [saiga](https://github.com/IlyaGusev/saiga)
- Tokenizer extension code: partially based on [Qwen](https://github.com/QwenLM/Qwen)
- Evaluation framework: [llmtf_open](https://github.com/RefalMachine/llmtf_open)
