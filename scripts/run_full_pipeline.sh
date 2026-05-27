#!/bin/bash
set -euo pipefail

# =============================================================================
# RuAdapt Full Pipeline Test Script
# =============================================================================
# Validates the entire pipeline: tokenizer extension → embedding init →
# token stats + trim → smart dataset → CPT + sub → eval + diagnostics
#
# Usage:
#   bash scripts/run_full_pipeline.sh --variant unigram128
#   bash scripts/run_full_pipeline.sh --variant bpe64
#   bash scripts/run_full_pipeline.sh --variant both
# =============================================================================

# --- Defaults ---
BASE_MODEL="/workdir/models/Qwen3.5-2B-Base"
DONOR_TOKENIZERS_DIR="/workdir/devel/ruadapt/ruadapt/tokenization/hf_tokenizers"
CORPUS_1M="/workdir/tokenizer_init_research/data/train_part1_1kk.json"
CORPUS_PART1="/shared/data/data/pre-train/darulm_25_03_25/train_part1.json"
CORPUS_PART2="/shared/data/data/pre-train/darulm_25_03_25/train_part2.json"
WORK_DIR="/workdir/pipeline_test"
VARIANT="both"
FREEZE_IDX=""  # auto-detect if empty
K_COVERAGE=100
TRIM_K=50
CPT_STEPS=5000
NUM_PROC=16
CUDA_DEVICE="0"

# --- Parse args ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --variant) VARIANT="$2"; shift ;;
        --base_model) BASE_MODEL="$2"; shift ;;
        --work_dir) WORK_DIR="$2"; shift ;;
        --freeze_idx) FREEZE_IDX="$2"; shift ;;
        --k_coverage) K_COVERAGE="$2"; shift ;;
        --trim_k) TRIM_K="$2"; shift ;;
        --cpt_steps) CPT_STEPS="$2"; shift ;;
        --cuda) CUDA_DEVICE="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

# --- Tokenizer configs ---
declare -A TOK_TYPES
TOK_TYPES[unigram128]="$DONOR_TOKENIZERS_DIR/darulm_20_05_24_part1-2_128000_unigram_hf"
TOK_TYPES[bpe64]="$DONOR_TOKENIZERS_DIR/darulm_20_05_24_part1-2_64000_bpe_hf"

declare -A TOK_EXTRACT_TYPES
TOK_EXTRACT_TYPES[unigram128]="unigram"
TOK_EXTRACT_TYPES[bpe64]="bpe"

# --- Helper functions ---
log() {
    echo ""
    echo "================================================================"
    echo " $1"
    echo "================================================================"
}

check_file() {
    if [ ! -f "$1" ] && [ ! -d "$1" ]; then
        echo "ERROR: Missing: $1"
        exit 1
    fi
}

# --- Auto-detect freeze_idx ---
if [ -z "$FREEZE_IDX" ]; then
    log "Auto-detecting freeze_idx from base model"
    FREEZE_IDX=$(python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('$BASE_MODEL', trust_remote_code=True)
print(len(tok))
" | tail -1)
    echo "freeze_idx = $FREEZE_IDX (base model vocab size)"
fi

# --- Variant list ---
if [ "$VARIANT" = "both" ]; then
    VARIANTS=(unigram128 bpe64)
else
    VARIANTS=($VARIANT)
fi

for VAR in "${VARIANTS[@]}"; do
    DONOR_TOK="${TOK_TYPES[$VAR]}"
    EXTRACT_TYPE="${TOK_EXTRACT_TYPES[$VAR]}"
    VAR_DIR="$WORK_DIR/$VAR"

    echo ""
    echo "########################################################"
    echo "# VARIANT: $VAR"
    echo "# Donor: $DONOR_TOK"
    echo "# Work dir: $VAR_DIR"
    echo "########################################################"

    check_file "$DONOR_TOK"
    check_file "$BASE_MODEL"
    check_file "$CORPUS_1M"

    # =========================================================================
    # STEP 1: Build Extended Tokenizer
    # =========================================================================
    log "STEP 1: Build Extended Tokenizer ($VAR)"
    TOK_DIR="$VAR_DIR/tokenizer"
    mkdir -p "$TOK_DIR"

    # 1a. Extract vocab from donor tokenizer
    echo "[1a] Extracting vocab from donor tokenizer..."
    python -m ruadapt.tokenization.convert cli_extract_vocab \
        --tokenizer_path "$DONOR_TOK" \
        --output_path "$TOK_DIR/vocab_freq.txt" \
        --type "$EXTRACT_TYPE" \
        --only_ru \
        --min_len 4

    # 1b. Extend base tokenizer
    echo "[1b] Extending base tokenizer..."
    python -m ruadapt.tokenization.core \
        --base_tokenizer "$BASE_MODEL" \
        --vocab "$TOK_DIR/vocab_freq.txt" \
        --output_dir "$TOK_DIR"

    echo "Extended tokenizer saved to: $TOK_DIR"
    echo "New vocab size: $(python -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('$TOK_DIR'); print(len(t))")"

    # =========================================================================
    # STEP 2a: Mean Init
    # =========================================================================
    log "STEP 2a: Mean Init ($VAR)"
    MODEL_MEAN="$VAR_DIR/model_mean"
    mkdir -p "$MODEL_MEAN"

    python -m ruadapt.tokenization.cli \
        --model_name_or_path "$BASE_MODEL" \
        --new_tokenizer_path "$TOK_DIR" \
        --output_path "$MODEL_MEAN" \
        --mode mean

    echo "Mean init model saved to: $MODEL_MEAN"

    # =========================================================================
    # STEP 2b: RelDist Init (head training + application)
    # =========================================================================
    log "STEP 2b: RelDist Init ($VAR)"
    CACHE_DIR="$VAR_DIR/head/cache"
    HEAD_DIR="$VAR_DIR/head/results"
    MODEL_RELDIST="$VAR_DIR/model_reldist"
    mkdir -p "$CACHE_DIR" "$HEAD_DIR" "$MODEL_RELDIST"

    # 2b.1: Build BPE dataset for head training
    echo "[2b.1] Building BPE dataset for head training..."
    # NOTE: build_bpe_dataset has hardcoded model path — may need adjustment
    python -m ruadapt.initialization.data.build_bpe_dataset \
        --min_len 4 --trials 1000 --min_count 50

    # 2b.2: Precompute hidden states
    echo "[2b.2] Precomputing hidden states..."
    python -m ruadapt.initialization.cache.precompute \
        --model-path "$BASE_MODEL" \
        --train-file data/train_bpe.json \
        --val-file data/val_bpe.json \
        --cache-dir "$CACHE_DIR" \
        --chunk-size 10000 \
        --batch-size 512

    # 2b.3: Train head with reldist loss
    echo "[2b.3] Training head (reldist_mse_cosine, rdr=0.3)..."
    python -m ruadapt.initialization.head.train \
        --pooling attention \
        --loss-type reldist_mse_cosine \
        --reldist-ratio 0.3 \
        --epochs 5 \
        --cache-dir "$CACHE_DIR" \
        --model-path "$BASE_MODEL" \
        --results-dir "$HEAD_DIR"

    HEAD_PT="$HEAD_DIR/reldist_mse_cosine_head_final.pt"
    if [ ! -f "$HEAD_PT" ]; then
        # Try to find any head_final.pt
        HEAD_PT=$(find "$HEAD_DIR" -name "*head_final.pt" | head -1)
    fi
    echo "Trained head: $HEAD_PT"

    # 2b.4: Apply head-initialized embeddings
    # TODO: Current cli.py doesn't support --mode mlp. This step will fail
    # until replace_tokenizer_batched.py is migrated.
    echo "[2b.4] Applying head-initialized embeddings..."
    python -m ruadapt.tokenization.cli \
        --model_name_or_path "$BASE_MODEL" \
        --new_tokenizer_path "$TOK_DIR" \
        --output_path "$MODEL_RELDIST" \
        --mode mlp \
        --head_path "$HEAD_PT" \
        --pooling attention || {
            echo "WARNING: --mode mlp not yet implemented in cli.py. Skipping reldist init."
            echo "TODO: Migrate replace_tokenizer_batched.py from tokenizer_init_research."
        }

    # =========================================================================
    # STEP 3: Token Statistics + Trim Model
    # =========================================================================
    log "STEP 3: Token Statistics + Trim ($VAR)"
    TRIM_DIR="$VAR_DIR/trim"
    MODEL_TRIMMED="$VAR_DIR/model_trimmed"
    mkdir -p "$TRIM_DIR" "$MODEL_TRIMMED"

    # Use mean model as source for trim (reldist may have failed)
    MODEL_SRC="$MODEL_MEAN"
    if [ -d "$MODEL_RELDIST" ] && [ -f "$MODEL_RELDIST/config.json" ]; then
        MODEL_SRC="$MODEL_RELDIST"
    fi

    # 3.1: Trim tokenizer
    echo "[3.1] Trimming tokenizer (K=$TRIM_K, 1M docs)..."
    python scripts/trim_tokenizer.py \
        --model_path "$MODEL_SRC" \
        --data_path "$CORPUS_1M" \
        --output_dir "$TRIM_DIR" \
        --freeze_idx "$FREEZE_IDX" \
        --K "$TRIM_K" \
        --num_proc "$NUM_PROC" \
        --max_samples 1000000

    # 3.2: Trim model
    echo "[3.2] Trimming model (resizing embeddings)..."
    python scripts/trim_model.py \
        --model_path "$MODEL_SRC" \
        --trim_dir "$TRIM_DIR" \
        --output_dir "$MODEL_TRIMMED" \
        --dtype bfloat16

    echo "Trimmed model saved to: $MODEL_TRIMMED"

    # =========================================================================
    # STEP 4: Build Smart K=100 Dataset
    # =========================================================================
    log "STEP 4: Build Smart K=$K_COVERAGE Dataset ($VAR)"
    SMART_DATASET="$VAR_DIR/smart_k${K_COVERAGE}.json"
    mkdir -p "$VAR_DIR"

    python -m ruadapt.initialization.data.build_smart_dataset \
        --tokenizer_path "$MODEL_TRIMMED" \
        --output_path "$SMART_DATASET" \
        --base_tokenizer_path "$BASE_MODEL" \
        --k_coverage "$K_COVERAGE" \
        --num_proc "$NUM_PROC" \
        --data_files "$CORPUS_PART1" "$CORPUS_PART2"

    echo "Smart dataset saved to: $SMART_DATASET"

    # =========================================================================
    # STEP 5: CPT + Sub Target (2 models)
    # =========================================================================
    log "STEP 5: CPT + Sub Target Training ($VAR)"

    # Determine freeze_idx for trimmed model
    TRIM_FREEZE_IDX=$(python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('$MODEL_TRIMMED', trust_remote_code=True)
# freeze_idx = number of frozen base tokens (before new tokens start)
# This should match the original freeze_idx
print($FREEZE_IDX)
")

    # 5.1: Standard CPT
    log "STEP 5.1: Standard CPT ($VAR)"
    MODEL_CPT="$VAR_DIR/cpt"
    CPT_CONFIG="$VAR_DIR/cpt_config.json"
    mkdir -p "$MODEL_CPT"

    cat > "$CPT_CONFIG" <<EOF
{
    "model": {
        "model_name_or_path": "$MODEL_TRIMMED",
        "torch_dtype": "bfloat16",
        "attn_implementation": "flash_attention_2"
    },
    "data": {
        "train_file": "$SMART_DATASET",
        "block_size": 512,
        "preprocessing_num_workers": $NUM_PROC,
        "max_text_length": 50000
    },
    "lora": {"peft": false},
    "freeze": {
        "strategy": "embed_only",
        "freeze_idx": $TRIM_FREEZE_IDX
    },
    "unified_dataset": {
        "natural_boundaries": true,
        "fragment_ratio": 0.0,
        "p_split": 0.3
    },
    "training": {
        "output_dir": "$MODEL_CPT",
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 32,
        "gradient_accumulation_steps": 2,
        "max_steps": $CPT_STEPS,
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "embed_weight_decay": 0.0,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "wsd_constant_part": 0.85,
        "logging_steps": 10,
        "eval_steps": 500,
        "eval_strategy": "steps",
        "save_steps": 1000,
        "save_strategy": "steps",
        "save_total_limit": 1,
        "bf16": true,
        "optim": "adamw_torch",
        "max_grad_norm": 1.0,
        "seed": 42,
        "gradient_checkpointing": false,
        "dataloader_num_workers": 4,
        "remove_unused_columns": false,
        "report_to": []
    },
    "dataset_factory": "ruadapt.training.datasets.unified_factory.UnifiedDatasetFactory",
    "collator_factory": "ruadapt.training.datasets.unified_factory.UnifiedCollatorFactory"
}
EOF

    python -m ruadapt.training.train --config "$CPT_CONFIG"

    # 5.2: Substitution Target Training (hybrid = targeted + random)
    log "STEP 5.2: Sub Target Training ($VAR)"
    MODEL_SUB="$VAR_DIR/sub"
    SUB_CONFIG="$VAR_DIR/sub_config.json"
    mkdir -p "$MODEL_SUB"

    cat > "$SUB_CONFIG" <<EOF
{
    "model": {
        "model_name_or_path": "$MODEL_TRIMMED",
        "torch_dtype": "bfloat16",
        "attn_implementation": "flash_attention_2"
    },
    "data": {
        "train_file": "$SMART_DATASET",
        "block_size": 512,
        "preprocessing_num_workers": $NUM_PROC,
        "max_text_length": 50000
    },
    "lora": {"peft": false},
    "freeze": {
        "strategy": "embed_only",
        "freeze_idx": $TRIM_FREEZE_IDX
    },
    "unified_dataset": {
        "natural_boundaries": true,
        "fragment_ratio": 0.2,
        "p_split": 0.3,
        "substitution_method": "hybrid",
        "trim_dir": "$TRIM_DIR",
        "min_parent_freq_ratio": 3.0,
        "random_sub_ratio": 0.1
    },
    "training": {
        "output_dir": "$MODEL_SUB",
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 32,
        "gradient_accumulation_steps": 2,
        "max_steps": $CPT_STEPS,
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "embed_weight_decay": 0.0,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "wsd_constant_part": 0.85,
        "logging_steps": 10,
        "eval_steps": 500,
        "eval_strategy": "steps",
        "save_steps": 1000,
        "save_strategy": "steps",
        "save_total_limit": 1,
        "bf16": true,
        "optim": "adamw_torch",
        "max_grad_norm": 1.0,
        "seed": 42,
        "gradient_checkpointing": false,
        "dataloader_num_workers": 4,
        "remove_unused_columns": false,
        "report_to": []
    },
    "dataset_factory": "ruadapt.training.datasets.unified_factory.UnifiedDatasetFactory",
    "collator_factory": "ruadapt.training.datasets.unified_factory.UnifiedCollatorFactory"
}
EOF

    python -m ruadapt.training.train --config "$SUB_CONFIG"

    # =========================================================================
    # STEP 6: Fix Configs + Evaluate
    # =========================================================================
    log "STEP 6: Fix Configs + Evaluate ($VAR)"
    EVAL_DIR="$VAR_DIR/eval"
    mkdir -p "$EVAL_DIR"

    # 6.1: Fix VLM configs
    echo "[6.1] Fixing VLM configs..."
    python scripts/fix_config.py \
        --original "$BASE_MODEL" \
        --adapted "$MODEL_CPT" || echo "WARNING: fix_config failed for CPT model"

    python scripts/fix_config.py \
        --original "$BASE_MODEL" \
        --adapted "$MODEL_SUB" || echo "WARNING: fix_config failed for Sub model"

    # 6.2: Run llmtf evaluation
    echo "[6.2] Running llmtf evaluation..."
    python -m ruadapt.evaluation.runner \
        --model_path "$MODEL_CPT" \
        --tasks darumeru,rucola \
        --output_dir "$EVAL_DIR/cpt" || echo "WARNING: Evaluation failed for CPT model"

    python -m ruadapt.evaluation.runner \
        --model_path "$MODEL_SUB" \
        --tasks darumeru,rucola \
        --output_dir "$EVAL_DIR/sub" || echo "WARNING: Evaluation failed for Sub model"

    # =========================================================================
    # STEP 7: Tokenization Diagnostics + PPL Comparison
    # =========================================================================
    log "STEP 7: Diagnostics + PPL ($VAR)"
    DIAG_DIR="$VAR_DIR/diag"
    mkdir -p "$DIAG_DIR"

    # 7.1: Diagnostic report (base model)
    echo "[7.1] Diagnostic: base model..."
    python -m ruadapt.initialization.eval.diagnostic \
        --model_path "$BASE_MODEL" \
        --ppl_langs rus eng \
        --num_docs 200 \
        --output_report "$DIAG_DIR/base_diagnostic.md" || echo "WARNING: Diagnostic failed for base"

    # 7.2: Diagnostic report (mean init)
    echo "[7.2] Diagnostic: mean init..."
    python -m ruadapt.initialization.eval.diagnostic \
        --model_path "$MODEL_MEAN" \
        --ppl_langs rus eng \
        --num_docs 200 \
        --output_report "$DIAG_DIR/mean_diagnostic.md" || echo "WARNING: Diagnostic failed for mean"

    # 7.3: Diagnostic report (CPT model)
    echo "[7.3] Diagnostic: CPT model..."
    python -m ruadapt.initialization.eval.diagnostic \
        --model_path "$MODEL_CPT" \
        --ppl_langs rus eng \
        --num_docs 200 \
        --output_report "$DIAG_DIR/cpt_diagnostic.md" || echo "WARNING: Diagnostic failed for CPT"

    # 7.4: Diagnostic report (Sub model)
    echo "[7.4] Diagnostic: Sub model..."
    python -m ruadapt.initialization.eval.diagnostic \
        --model_path "$MODEL_SUB" \
        --ppl_langs rus eng \
        --num_docs 200 \
        --output_report "$DIAG_DIR/sub_diagnostic.md" || echo "WARNING: Diagnostic failed for Sub"

    # 7.5: PPL comparison
    echo "[7.5] PPL comparison..."
    for MODEL_NAME in mean cpt sub; do
        MODEL_PATH="$VAR_DIR/model_${MODEL_NAME}"
        if [ "$MODEL_NAME" = "cpt" ]; then MODEL_PATH="$MODEL_CPT"; fi
        if [ "$MODEL_NAME" = "sub" ]; then MODEL_PATH="$MODEL_SUB"; fi

        python -m ruadapt.initialization.eval.ppl \
            --model_path "$MODEL_PATH" \
            --data_path data/rus.json \
            --num_docs 1000 --max_tokens 512 \
            > "$DIAG_DIR/ppl_${MODEL_NAME}.txt" 2>&1 || echo "WARNING: PPL failed for $MODEL_NAME"
    done

    echo ""
    echo "================================================"
    echo " VARIANT $VAR COMPLETE"
    echo "================================================"
    echo " Results in: $VAR_DIR/"
    echo "   tokenizer/    — extended tokenizer"
    echo "   model_mean/   — mean init model"
    echo "   model_reldist/ — reldist init model (if mlp mode worked)"
    echo "   head/         — trained head + cache"
    echo "   trim/         — trim stats + trimmed tokenizer"
    echo "   model_trimmed/ — trimmed model"
    echo "   smart_k${K_COVERAGE}.json — smart dataset"
    echo "   cpt/          — CPT model"
    echo "   sub/          — Sub target model"
    echo "   eval/         — llmtf evaluation results"
    echo "   diag/         — diagnostic reports + PPL"
    echo "================================================"

done

log "ALL VARIANTS COMPLETE"
echo "Results in: $WORK_DIR/"
