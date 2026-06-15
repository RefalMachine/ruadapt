"""Thin wrapper around ruadapt.tokenization.trim for backward compatibility.

Core logic has been moved to ruadapt/tokenization/trim.py.
This script preserves the original CLI interface.

Usage:
    python -m scripts.trim_tokenizer \
        --model_path /path/to/model \
        --data_path /path/to/train.jsonl \
        --output_dir /path/to/output \
        --freeze_idx 248044 \
        --K 50 \
        --num_proc 16
"""

from ruadapt.tokenization.trim import main

if __name__ == "__main__":
    main()
