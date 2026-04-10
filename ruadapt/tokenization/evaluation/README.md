# Evaluation Tools

This directory contains scripts for benchmarking and validating extended tokenizers and language models. It is designed to be integrated into the `ruadapt` pipeline.

## Core Diagnostic Pipeline

The main entry point for evaluating a model's tokenization efficiency and perplexity is `run_diagnostic.py`.

### `run_diagnostic.py`
Generates a comprehensive Markdown report containing Tokenization Efficiency (Chars per Token) across multiple languages, and Corpus Perplexity (PPL) for specified languages.
- **Usage:**
  ```bash
  python run_diagnostic.py \
      --model_path /path/to/model \
      --data_dir data \
      --ppl_langs rus eng \
      --output_report report.md
  ```
- **Dependencies:** Imports core logic from `evaluate_tokenizer.py` and `evaluate_ppl.py`.

---

## Modular Evaluation Scripts

These scripts can be run standalone or imported as modules:

### `evaluate_ppl.py`
Computes the true Corpus Perplexity (PPL) using the model's forward pass. It handles truncation and batching correctly.
- **Usage:** `python evaluate_ppl.py --model_path <path> --data_path data/rus.json`

### `evaluate_tokenizer.py`
Calculates Characters Per Token (CPT) for all JSON files in the `--data_dir`. Higher CPT means the tokenizer compresses the language more efficiently.
- **Usage:** `python evaluate_tokenizer.py --tokenizer_path <path>`

### `test_token_conversion.py`
A crucial unit-test for the byte-level BPE conversion logic. It simulates the `ruadapt` embedding initialization logic (extracting string representations of BPE bytes like `ĠÐ±` and re-tokenizing them).
- **Usage:** `python test_token_conversion.py --base_tokenizer <path> --new_tokenizer <path>`
- *Note: Expects 0 mismatches on successful runs. Exits with code 1 if errors are found.*

### `compare_tokenizers.py`
Directly compares two different tokenizers (base vs new) side-by-side to calculate the exact percentage of token savings across text datasets. Essential for quick sanity checks on tokenizer compression without running a full model forward pass.
- **Usage:** `python compare_tokenizers.py --base_tokenizer <path> --new_tokenizer <path>`

### `download_data.py`

---

## Data Format

Scripts expect data in `evaluation/data/` as JSON arrays containing objects with a `"text"` field:
```json
[
  {"text": "Sample document 1"},
  {"text": "Sample document 2"}
]
```