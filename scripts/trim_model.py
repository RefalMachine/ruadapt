"""Thin wrapper around ruadapt.tokenization.trim_model for backward compatibility.

Core logic has been moved to ruadapt/tokenization/trim_model.py.
This script preserves the original CLI interface.

Usage:
    python -m scripts.trim_model \
        --model_path /path/to/original_model \
        --trim_dir /path/to/trim_output \
        --output_dir /path/to/trimmed_model \
        --dtype bfloat16
"""

from ruadapt.tokenization.trim_model import main

if __name__ == "__main__":
    main()
