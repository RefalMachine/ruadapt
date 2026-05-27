# Pipeline Test Plan

## Overview

End-to-end validation of the entire RuAdapt pipeline after migration. Tests two tokenization variants (unigram-128, bpe-64) through the full path: tokenizer extension → embedding init → token stats + trim → smart dataset → CPT + sub → eval + diagnostics.

**Models to test on:**
- Qwen3.5-2B-Base (primary, VLM)
- Qwen3-8B-Base (secondary, text-only) — TODO: add after Qwen3.5 path is validated

---

## Pipeline Steps (per tokenization variant)

### Step 1: Build Extended Tokenizer

Extract Russian tokens from donor tokenizer, inject into base tokenizer.

**Inputs:**
- Base model: `$BASE_MODEL` (e.g. `/workdir/models/Qwen3.5-2B-Base`)
- Donor tokenizer: trained tokenizer (unigram-128 or bpe-64)
- `--min_len 4` — minimum Cyrillic character length

**Command:**
```bash
python -m ruadapt.tokenization.convert cli_extract_vocab \
    --tokenizer_path $DONOR_TOK \
    --output_path $TOK_DIR/vocab_freq.txt \
    --type $TOK_TYPE \
    --only_ru \
    --min_len 4

python -m ruadapt.tokenization.core \
    --base_tokenizer $BASE_MODEL \
    --vocab $TOK_DIR/vocab_freq.txt \
    --output_dir $TOK_DIR
```

**Outputs:** `$TOK_DIR/` with `tokenizer.json`, `config.json`, `tokenizer_config.json`

**Verify:** tokenizer loads, vocab_size > base vocab_size

---

### Step 2a: Mean Init

Initialize new token embeddings as mean of subword decompositions.

**Command:**
```bash
python -m ruadapt.tokenization.cli \
    --model_name_or_path $BASE_MODEL \
    --new_tokenizer_path $TOK_DIR \
    --output_path $MODEL_MEAN \
    --mode mean
```

**Verify:** model loads, new tokens have non-zero embeddings, PPL not catastrophically broken

---

### Step 2b: RelDist Init (requires head training)

Train MLP head on precomputed hidden states, then use it to initialize embeddings.

#### 2b.1: Build BPE dataset for head training
```bash
python -m ruadapt.initialization.data.build_bpe_dataset \
    --min_len 4 --trials 1000 --min_count 50
```

#### 2b.2: Precompute hidden states
```bash
python -m ruadapt.initialization.cache.precompute \
    --model-path $BASE_MODEL \
    --train-file data/train_bpe.json \
    --val-file data/val_bpe.json \
    --cache-dir $CACHE_DIR \
    --chunk-size 10000 \
    --batch-size 512
```

#### 2b.3: Train head (reldist loss)
```bash
python -m ruadapt.initialization.head.train \
    --pooling attention \
    --loss-type reldist_mse_cosine \
    --reldist-ratio 0.3 \
    --epochs 5 \
    --cache-dir $CACHE_DIR \
    --model-path $BASE_MODEL \
    --results-dir $HEAD_DIR
```

#### 2b.4: Apply head-initialized embeddings
```bash
python -m ruadapt.tokenization.cli \
    --model_name_or_path $BASE_MODEL \
    --new_tokenizer_path $TOK_DIR \
    --output_path $MODEL_RELDIST \
    --mode mean
```
Note: `cli.py` currently only supports mean/wmean/random. For reldist init, need to either:
- Extend cli.py to accept `--head_path` and use head predictions, OR
- Use a separate script that loads head weights and applies them

**TODO**: Check if head application is already handled elsewhere, or if cli.py needs extension.

**Verify:** model loads, embeddings match head predictions (spot-check cosine sim)

---

### Step 3: Token Statistics + Trim Model

Compute token frequencies on 1M document corpus, cascade-trim rare tokens.

#### 3.1: Trim tokenizer
```bash
python scripts/trim_tokenizer.py \
    --model_path $MODEL_SRC \
    --data_path $CORPUS_1M \
    --output_dir $TRIM_DIR \
    --freeze_idx $FREEZE_IDX \
    --K 50 \
    --num_proc 16 \
    --max_samples 1000000
```

#### 3.2: Trim model (resize embeddings)
```bash
python scripts/trim_model.py \
    --model_path $MODEL_SRC \
    --trim_dir $TRIM_DIR \
    --output_dir $MODEL_TRIMMED \
    --dtype bfloat16
```

**Verify:** trimmed model loads, vocab_size < original, no OOV crash on sample text

---

### Step 4: Build Smart K=100 Dataset

Using the trimmed model's tokenizer, build a dataset where every new token appears ≥100 times.

```bash
python -m ruadapt.initialization.data.build_smart_dataset \
    --tokenizer_path $MODEL_TRIMMED \
    --output_path $SMART_DATASET \
    --base_tokenizer_path $BASE_MODEL \
    --k_coverage 100 \
    --num_proc 16 \
    --data_files $CORPUS_FILES
```

**Verify:** all new tokens appear ≥100 times in output, dataset size is reasonable

---

### Step 5: CPT + Sub Target (2 models)

#### 5.1: Standard CPT
```bash
python -m ruadapt.training.train --config $CPT_CONFIG
```

Config template:
```json
{
  "model": {
    "model_name_or_path": "$MODEL_TRIMMED",
    "torch_dtype": "bfloat16",
    "attn_implementation": "flash_attention_2"
  },
  "data": {
    "train_file": "$SMART_DATASET",
    "block_size": 512,
    "preprocessing_num_workers": 16
  },
  "lora": {"peft": false},
  "freeze": {
    "strategy": "embed_only",
    "freeze_idx": $FREEZE_IDX
  },
  "unified_dataset": {
    "natural_boundaries": true,
    "fragment_ratio": 0.0,
    "p_split": 0.3
  },
  "training": {
    "output_dir": "$MODEL_CPT",
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "max_steps": 5000,
    "learning_rate": 3e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "eval_steps": 500,
    "save_steps": 1000,
    "bf16": true,
    "seed": 42,
    "report_to": []
  },
  "dataset_factory": "ruadapt.training.datasets.unified_factory.UnifiedDatasetFactory",
  "collator_factory": "ruadapt.training.datasets.unified_factory.UnifiedCollatorFactory"
}
```

#### 5.2: Substitution Target Training
Same config but with substitution enabled (hybrid = targeted + random):
```json
{
  "unified_dataset": {
    "natural_boundaries": true,
    "fragment_ratio": 0.2,
    "p_split": 0.3,
    "substitution_method": "hybrid",
    "trim_dir": "$TRIM_DIR",
    "min_parent_freq_ratio": 3.0,
    "random_sub_ratio": 0.1
  }
}
```

Hybrid method: first applies targeted substitution to pool tokens (split parents with rare children, probability = `fragment_ratio`), then applies random BPE-Dropout to remaining eligible non-pool tokens (probability = `fragment_ratio * random_sub_ratio` ≈ 0.2 * 0.1 = 2%). Pool tokens are never touched by the random phase.

**Verify:** both models train without errors, loss decreases, checkpoints saved

---

### Step 6: Fix Configs + Evaluate

#### 6.1: Fix VLM configs (Qwen3.5 only)
```bash
python scripts/fix_config.py \
    --original $BASE_MODEL \
    --adapted $MODEL_CPT

python scripts/fix_config.py \
    --original $BASE_MODEL \
    --adapted $MODEL_SUB
```

#### 6.2: Run llmtf evaluation
```bash
python -m ruadapt.evaluation.runner \
    --model_path $MODEL_CPT \
    --tasks darumeru,rucola \
    --output_dir $EVAL_DIR/cpt

python -m ruadapt.evaluation.runner \
    --model_path $MODEL_SUB \
    --tasks darumeru,rucola \
    --output_dir $EVAL_DIR/sub
```

**Verify:** evaluation completes, results are reasonable (not random)

---

### Step 7: Tokenization Diagnostics + PPL Comparison

#### 7.1: Diagnostic report (base model, pre-CPT)
```bash
python -m ruadapt.initialization.eval.diagnostic \
    --model_path $BASE_MODEL \
    --ppl_langs rus eng \
    --num_docs 200 \
    --output_report $DIAG_DIR/base_diagnostic.md
```

#### 7.2: Diagnostic report (CPT model)
```bash
python -m ruadapt.initialization.eval.diagnostic \
    --model_path $MODEL_CPT \
    --ppl_langs rus eng \
    --num_docs 200 \
    --output_report $DIAG_DIR/cpt_diagnostic.md
```

#### 7.3: Diagnostic report (Sub model)
```bash
python -m ruadapt.initialization.eval.diagnostic \
    --model_path $MODEL_SUB \
    --ppl_langs rus eng \
    --num_docs 200 \
    --output_report $DIAG_DIR/sub_diagnostic.md
```

#### 7.4: PPL comparison (standalone)
```bash
# Pre-CPT PPL
python -m ruadapt.initialization.eval.ppl \
    --model_path $MODEL_MEAN \
    --data_path data/rus.json \
    --num_docs 1000 --max_tokens 512

# Post-CPT PPL
python -m ruadapt.initialization.eval.ppl \
    --model_path $MODEL_CPT \
    --data_path data/rus.json \
    --num_docs 1000 --max_tokens 512

# Post-Sub PPL
python -m ruadapt.initialization.eval.ppl \
    --model_path $MODEL_SUB \
    --data_path data/rus.json \
    --num_docs 1000 --max_tokens 512
```

**Verify:** Russian PPL improves post-CPT vs pre-CPT. English PPL doesn't catastrophically regress.

---

## Directory Layout

```
$WORK_DIR/
├── unigram128/
│   ├── tokenizer/           # Step 1: extended tokenizer
│   ├── model_mean/          # Step 2a: mean init model
│   ├── head/                # Step 2b: trained head
│   │   ├── cache/
│   │   └── results/
│   ├── model_reldist/       # Step 2b: reldist init model
│   ├── trim/                # Step 3: trim output
│   ├── model_trimmed/       # Step 3: trimmed model
│   ├── smart_k100.json      # Step 4: smart dataset
│   ├── cpt/                 # Step 5.1: CPT model
│   ├── sub/                 # Step 5.2: Sub model
│   ├── eval/                # Step 6: evaluation results
│   └── diag/                # Step 7: diagnostic reports
├── bpe64/
│   └── (same structure)
└── logs/
```

---

## Open Questions / TODOs

### Critical: `--mode mlp` missing from cli.py

The old `run_replace_tokenizer.py` (from tokenizer_init_research) supported `--mode mlp` with `--head_path` and `--pooling` args via `replace_tokenizer_batched.py`. This file was NOT migrated. The current `cli.py` only supports mean/wmean/random.

**Action needed**: Migrate `replace_tokenizer_batched.py` (377 lines) from tokenizer_init_research into `ruadapt/tokenization/replace_batched.py`, and extend `cli.py` with `--mode mlp`, `--head_path`, `--pooling`, `--batch_size` args.

Source: `/workdir/tokenizer_init_research/ruadapt/ruadapt/tokenization/replace_tokenizer_batched.py`

### Resolved

- **Donor tokenizers**: `/workdir/devel/ruadapt/ruadapt/tokenization/hf_tokenizers/`
  - unigram-128: `darulm_20_05_24_part1-2_128000_unigram_hf`
  - bpe-64: `darulm_20_05_24_part1-2_64000_bpe_hf`
- **Corpus for 1M**: `/workdir/tokenizer_init_research/data/train_part1_1kk.json` (1M lines)
- **Full corpus for smart**: `/shared/data/data/pre-train/darulm_25_03_25/train_part1.json` + `train_part2.json`
- **freeze_idx**: Dynamic. For Qwen3.5-2B-Base with these tokenizers, verify it equals 248044.
- **Base model**: `/workdir/models/Qwen3.5-2B-Base`
