"""Resize model embeddings/lm_head after tokenizer trimming.

Takes the original model and trim results, produces a new model with:
- Resized embed_tokens (rows reordered & shrunk per old_to_new mapping)
- Resized lm_head (tied or separate)
- Updated config (vocab_size, special token IDs)
- Saved tokenizer from trim output

CLI entrypoint:
    python -m ruadapt.tokenization.trim_model \
        --model_path /path/to/original_model \
        --trim_dir /path/to/trim_output \
        --output_dir /path/to/trimmed_model \
        --dtype bfloat16
"""

import argparse
import json
import os
import shutil
from typing import Dict

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def build_new_embeddings(
    old_weight: torch.Tensor,
    old_to_new: dict,
    new_vocab_size: int,
) -> torch.Tensor:
    """Build new embedding matrix by copying rows per old_to_new mapping.

    Tokens not in old_to_new (removed) get zero vectors.

    Args:
        old_weight: [old_vocab_size, hidden_dim] embedding matrix.
        old_to_new: dict mapping old_token_id (int) -> new_token_id (int).
        new_vocab_size: total rows in the new embedding matrix.

    Returns:
        new_weight: [new_vocab_size, hidden_dim] tensor.
    """
    hidden_dim = old_weight.shape[1]
    device = old_weight.device
    dtype = old_weight.dtype

    new_weight = torch.zeros(new_vocab_size, hidden_dim, device=device, dtype=dtype)

    for old_id_str, new_id in old_to_new.items():
        old_id = int(old_id_str)
        if 0 <= old_id < old_weight.shape[0] and 0 <= new_id < new_vocab_size:
            new_weight[new_id] = old_weight[old_id]

    return new_weight


def trim_model(
    model_path: str,
    trim_dir: str,
    output_dir: str,
    dtype: str = "bfloat16",
) -> None:
    """Resize model embeddings after tokenizer trimming.

    Args:
        model_path: Path to the original model.
        trim_dir: Path to trim_tokenizer output directory.
        output_dir: Output directory for the trimmed model.
        dtype: Model dtype (bfloat16, float16, float32).
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load trim results ----
    print(f"Loading trim results from {trim_dir}...")
    with open(os.path.join(trim_dir, "token_id_mapping.json")) as f:
        mapping = json.load(f)

    old_to_new = mapping["old_to_new"]
    old_vocab_size = mapping["old_vocab_size"]
    new_vocab_size = mapping["new_vocab_size"]
    removed_count = mapping["removed_count"]

    with open(os.path.join(trim_dir, "tokenizer.json")) as f:
        tok_data = json.load(f)
    n_special = len(tok_data.get("added_tokens", []))
    total_new_vocab = new_vocab_size + n_special

    print(f"  Old vocab: {old_vocab_size}, New vocab: {total_new_vocab} "
          f"({new_vocab_size} regular + {n_special} special)")
    print(f"  Removed: {removed_count}")

    # ---- Load original model ----
    print(f"Loading model from {model_path}...")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    print(f"  Model loaded, vocab_size in config: {hf_config.vocab_size}")

    # ---- Identify embedding modules ----
    embed_tokens = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    tied = hf_config.tie_word_embeddings

    print(f"  tie_word_embeddings: {tied}")
    print(f"  embed_tokens shape: {list(embed_tokens.weight.shape)}")
    if lm_head is not None:
        print(f"  lm_head shape: {list(lm_head.weight.shape)}")
    else:
        print(f"  lm_head: None (tied)")

    old_embed_weight = embed_tokens.weight.data.clone()

    old_lm_weight = None
    if lm_head is not None and not tied:
        old_lm_weight = lm_head.weight.data.clone()

    # ---- Build new embeddings ----
    print("Building new embed_tokens...")
    new_embed_weight = build_new_embeddings(old_embed_weight, old_to_new, total_new_vocab)

    # ---- Apply new embeddings ----
    model.resize_token_embeddings(total_new_vocab)

    embed_tokens = model.get_input_embeddings()
    embed_tokens.weight.data.copy_(new_embed_weight)
    print(f"  New embed_tokens shape: {list(embed_tokens.weight.shape)}")

    # ---- Handle lm_head ----
    if tied:
        lm_head = model.get_output_embeddings()
        if lm_head is not None:
            assert lm_head.weight.data_ptr() == embed_tokens.weight.data_ptr(), \
                "Expected tied lm_head and embed_tokens to share weights"
        print("  lm_head: tied to embed_tokens (OK)")
    else:
        print("Building new lm_head...")
        new_lm_weight = build_new_embeddings(old_lm_weight, old_to_new, total_new_vocab)
        lm_head = model.get_output_embeddings()
        if lm_head is not None:
            lm_head.weight.data.copy_(new_lm_weight)
            print(f"  New lm_head shape: {list(lm_head.weight.shape)}")
        else:
            print("  WARNING: lm_head is None but tie_word_embeddings=False")

    # ---- Update config ----
    print("Updating config...")
    hf_config.vocab_size = total_new_vocab

    if hasattr(hf_config, "text_config") and hf_config.text_config is not None:
        hf_config.text_config.vocab_size = total_new_vocab
        print(f"  text_config.vocab_size: {total_new_vocab}")

    added = tok_data.get("added_tokens", [])
    added_by_content = {e["content"]: e["id"] for e in added}

    field_to_attr = {
        "bos_token_id": "bos_token",
        "eos_token_id": "eos_token",
        "pad_token_id": "pad_token",
        "unk_token_id": "unk_token",
        "cls_token_id": "cls_token",
        "sep_token_id": "sep_token",
        "mask_token_id": "mask_token",
    }

    tok_config_path = os.path.join(trim_dir, "tokenizer_config.json")
    tok_config = {}
    if os.path.isfile(tok_config_path):
        with open(tok_config_path) as f:
            tok_config = json.load(f)

    for field, attr in field_to_attr.items():
        new_id = None
        if attr in tok_config and isinstance(tok_config[attr], dict):
            content = tok_config[attr].get("content")
            if content and content in added_by_content:
                new_id = added_by_content[content]
        elif attr in tok_config and isinstance(tok_config[attr], str):
            if tok_config[attr] in added_by_content:
                new_id = added_by_content[tok_config[attr]]
        if new_id is None:
            old_id = getattr(hf_config, field, None)
            if old_id is not None and str(old_id) in old_to_new:
                new_id = old_to_new[str(old_id)]

        if new_id is not None:
            setattr(hf_config, field, new_id)
            if hasattr(hf_config, "text_config") and hf_config.text_config is not None:
                setattr(hf_config.text_config, field, new_id)
            print(f"  {field}: {new_id}")

    def _remap_token_ids(cfg, prefix=""):
        for key in list(vars(cfg).keys()):
            if "token_id" in key and key not in field_to_attr:
                old_val = getattr(cfg, key, None)
                if old_val is not None and isinstance(old_val, int) and str(old_val) in old_to_new:
                    new_val = old_to_new[str(old_val)]
                    setattr(cfg, key, new_val)
                    print(f"  {prefix}{key}: {old_val} -> {new_val}")

    _remap_token_ids(hf_config)
    if hasattr(hf_config, "text_config") and hf_config.text_config is not None:
        _remap_token_ids(hf_config.text_config, "text_config.")

    model.config = hf_config

    # ---- Verify ----
    final_embed = model.get_input_embeddings()
    final_lm = model.get_output_embeddings()
    print(f"\n  Final embed_tokens: {list(final_embed.weight.shape)}")
    if final_lm is not None:
        print(f"  Final lm_head: {list(final_lm.weight.shape)}")
    print(f"  Config vocab_size: {model.config.vocab_size}")

    tok = AutoTokenizer.from_pretrained(trim_dir, trust_remote_code=True)
    test = tok("Привет мир!", add_special_tokens=False)
    max_id = max(test["input_ids"])
    assert max_id < total_new_vocab, f"Token ID {max_id} >= vocab_size {total_new_vocab}!"
    print(f"  Tokenization test: OK (max_id={max_id})")

    # ---- Save ----
    print(f"\nSaving to {output_dir}...")
    model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)

    for fname in ["special_tokens_map.json"]:
        src = os.path.join(trim_dir, fname)
        if os.path.isfile(src) and not os.path.isfile(os.path.join(output_dir, fname)):
            shutil.copy2(src, os.path.join(output_dir, fname))
            print(f"  Copied {fname}")

    print(f"\nDone! Trimmed model saved to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Resize model after tokenizer trimming")
    parser.add_argument("--model_path", type=str, required=True, help="Path to original model")
    parser.add_argument("--trim_dir", type=str, required=True, help="Path to trim_tokenizer output")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for trimmed model")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Model dtype (bfloat16, float16, float32)")
    args = parser.parse_args()

    trim_model(
        model_path=args.model_path,
        trim_dir=args.trim_dir,
        output_dir=args.output_dir,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
