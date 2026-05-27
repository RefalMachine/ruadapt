#!/bin/bash

# Скрипт примеров запуска пайплайна расширения токенизатора
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIAG_SCRIPT="$DIR/ruadapt/tokenization/evaluation/run_diagnostic.py"

# ==============================================================================
# Вариант 1: VLM модель Qwen3.5-2B-Base
# ==============================================================================
echo "=============================================================================="
echo "Running Example 1: VLM Model (Qwen3.5-2B-Base)"
echo "=============================================================================="

echo -e "\n[Pre-eval] Evaluating Base Model (Qwen3.5-2B-Base)..."
python "$DIAG_SCRIPT" --model_path "/workdir/models/Qwen3.5-2B-Base" --ppl_langs rus eng --output_report "/workdir/models/Qwen3.5-2B-Base/tok_report.md"
cat /workdir/models/Qwen3.5-2B-Base/tok_report.md

bash "$DIR/run_extension_pipeline.sh" \
    --donor_tokenizer "$DIR/ruadapt/tokenization/hf_tokenizers/darulm_20_05_24_part1-2_128000_unigram_hf" \
    --base_model "/workdir/models/Qwen3.5-2B-Base" \
    --type "unigram" \
    --top_k "64000" \
    --init_mode "mean" \
    --output_dir "/workdir/models/RuadaptQwen3.5-2B-Base-Example"

echo -e "\n[Post-eval] Evaluating Extended Model (RuadaptQwen3.5-2B-Base-Example)..."
python "$DIAG_SCRIPT" --model_path "/workdir/models/RuadaptQwen3.5-2B-Base-Example" --ppl_langs rus eng --output_report "/workdir/models/RuadaptQwen3.5-2B-Base-Example/tok_report.md"
cat /workdir/models/RuadaptQwen3.5-2B-Base-Example/tok_report.md

# ==============================================================================
# Вариант 2: Текстовая модель Qwen/Qwen3-0.6B-Base
# ==============================================================================
echo -e "\n\n=============================================================================="
echo "Running Example 2: Text Model (Qwen3-0.6B-Base)"
echo "=============================================================================="

echo -e "\n[Pre-eval] Evaluating Base Model (Qwen3-0.6B-Base)..."
python "$DIAG_SCRIPT" --model_path "/workdir/models/Qwen3-0.6B-Base" --ppl_langs rus eng --output_report "/workdir/models/Qwen3-0.6B-Base/tok_report.md"
cat /workdir/models/Qwen3-0.6B-Base/tok_report.md

bash "$DIR/run_extension_pipeline.sh" \
    --donor_tokenizer "$DIR/ruadapt/tokenization/hf_tokenizers/darulm_20_05_24_part1-2_48000_unigram_hf" \
    --base_model "/workdir/models/Qwen3-0.6B-Base" \
    --type "unigram" \
    --init_mode "mean" \
    --min_len 0 \
    --output_dir "/workdir/models/RuadaptQwen3-0.6B-Base-Example"

echo -e "\n[Post-eval] Evaluating Extended Model (RuadaptQwen3-0.6B-Base-Example)..."
python "$DIAG_SCRIPT" --model_path "/workdir/models/RuadaptQwen3-0.6B-Base-Example" --ppl_langs rus eng --output_report "/workdir/models/RuadaptQwen3-0.6B-Base-Example/tok_report.md"
cat /workdir/models/RuadaptQwen3-0.6B-Base-Example/tok_report.md
