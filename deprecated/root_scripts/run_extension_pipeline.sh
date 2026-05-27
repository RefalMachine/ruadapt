#!/bin/bash
set -e # Останавливать скрипт при любой ошибке

# Значения по умолчанию
MIN_LEN=4
TYPE="bpe"
TOP_K=""
INIT_MODE="mean"
MULT=1.0

# Парсинг аргументов
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --donor_tokenizer) DONOR_TOK="$2"; shift ;;
        --base_model) BASE_MODEL="$2"; shift ;;
        --output_dir) OUT_DIR="$2"; shift ;;
        --type) TYPE="$2"; shift ;;
        --min_len) MIN_LEN="$2"; shift ;;
        --top_k) TOP_K="$2"; shift ;;
        --init_mode) INIT_MODE="$2"; shift ;;
        --mult) MULT="$2"; shift ;;
        --skip_model) SKIP_MODEL=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Проверка обязательных аргументов
if [ -z "$DONOR_TOK" ] || [ -z "$BASE_MODEL" ] || [ -z "$OUT_DIR" ]; then
    echo "Error: Missing required arguments."
    echo "Usage: $0 --donor_tokenizer <path> --base_model <path> --output_dir <path> [--type bpe|unigram] [--min_len <int>] [--skip_model]"
    exit 1
fi

echo "=== Pipeline Configuration ==="
echo "Donor Tokenizer: $DONOR_TOK"
echo "Base Model:      $BASE_MODEL"
echo "Output Dir:      $OUT_DIR"
echo "Type:            $TYPE"
echo "Min Length:      $MIN_LEN"
if [ "$SKIP_MODEL" != true ]; then
    echo "Init Mode:       $INIT_MODE"
    echo "Multiplier:      $MULT"
fi
echo "=============================="

# Создаем промежуточную директорию для токенизатора
TOK_DIR="${OUT_DIR}_tokenizer"
mkdir -p "$TOK_DIR"

VOCAB_FILE="$TOK_DIR/extracted_vocab_freq_min${MIN_LEN}.txt"

# Пути к скриптам относительно папки ruadapt
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

EXTRACT_CMD=(python "$DIR/ruadapt/tokenization/extract_vocab.py" --tokenizer_path "$DONOR_TOK" --output_path "$VOCAB_FILE" --type "$TYPE" --only_ru --min_len "$MIN_LEN")
if [ -n "$TOP_K" ]; then
    EXTRACT_CMD+=(--top_k "$TOP_K")
    echo "Top K:             $TOP_K"
fi

echo -e "\n[1/4] Extracting vocabulary..."
"${EXTRACT_CMD[@]}"

echo -e "\n[2/4] Extending base tokenizer..."
python "$DIR/ruadapt/tokenization/extend_tokenizer.py" \
    --base_tokenizer "$BASE_MODEL" \
    --vocab "$VOCAB_FILE" \
    --output_dir "$TOK_DIR"

echo -e "\n[3/4] Evaluating tokenizers..."
python "$DIR/ruadapt/tokenization/evaluation/compare_tokenizers.py" \
    --base_tokenizer "$BASE_MODEL" \
    --new_tokenizer "$TOK_DIR"

if [ "$SKIP_MODEL" != true ]; then
    echo -e "\n[4/4] Initializing model embeddings..."
    python -m ruadapt.tokenization.run_replace_tokenizer \
        --model_name_or_path "$BASE_MODEL" \
        --new_tokenizer_path "$TOK_DIR" \
        --output_path "$OUT_DIR" \
        --mode "$INIT_MODE" \
        --mult "$MULT"
    echo -e "\nPipeline completed! Final model saved to: $OUT_DIR"
else
    echo -e "\nPipeline completed! Tokenizer saved to: $TOK_DIR (model init skipped)"
fi
