#!/bin/bash
set -e

# 1. Update BPE merges and extra.json with proper start_id dynamically calculated from HF tokenizer
python3 ruadapt/ruadapt/tokenization/add_merges_fast.py \
  --input_path /workdir/models/Qwen3.5-2B-Base/tokenizer_new.tiktoken \
  --vocab_path /workdir/models/Qwen3.5-2B-Base/sp_tok_vocab_freq_u48_min4.txt \
  --output_path ./sp_tok_vocab_freq_u48_min4.tiktoken \
  --hf_tokenizer_dir /workdir/models/Qwen3.5-2B-Base

# 2. Inject new tokens and merges safely into the HuggingFace tokenizer
python3 -m ruadapt.ruadapt.tokenization.expand_tiktoken_save_hf \
  --tiktoken_base_path /workdir/models/Qwen3.5-2B-Base/tokenizer_new.tiktoken \
  --tiktoken_new_path ./sp_tok_vocab_freq_u48_min4.tiktoken \
  --output_dir /workdir/models/Qwen3.5-2B-Base/hf_tokenizer_u48_min4_v2 \
  --init_output_from /workdir/models/Qwen3.5-2B-Base

echo "Tokenization pipeline completed successfully."
