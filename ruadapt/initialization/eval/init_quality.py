#!/usr/bin/env python3
"""
Evaluate initialization quality by comparing new-token embeddings directly.

For mean/weighted_mean initialization: load the assembled model and the base model,
compare embeddings of new tokens (the ones added by the extended tokenizer).

No head, no cache, no hidden states needed — just two models.

Usage:
    python pipeline/training/eval_init_quality.py \
        --init-model /workdir/models/Qwen3.5-2B-Base_Eval_mean \
        --base-model /workdir/models/Qwen3.5-2B-Base \
        --ext-tokenizer /workdir/models/RuadaptQwen3.5-2B-Base-b64-minlen4-mean-v2_tokenizer \
        --passport data/token_passport.json \
        --output results/mean_init/per_token_init_metrics.json
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from ruadapt.utils.model_utils import load_causal_lm


def find_new_token_ids(base_model_path: str, ext_tokenizer_path: str):
    """Get IDs of tokens present in extended tokenizer but not in base."""
    from transformers import AutoTokenizer
    base_tok = AutoTokenizer.from_pretrained(base_model_path)
    ext_tok = AutoTokenizer.from_pretrained(ext_tokenizer_path)
    base_vocab = set(base_tok.get_vocab().keys())
    ext_vocab = set(ext_tok.get_vocab().keys())
    added = ext_vocab - base_vocab
    ext_full = ext_tok.get_vocab()
    return sorted([ext_full[tok] for tok in added])


def compute_init_metrics(init_embs: torch.Tensor, base_embs: torch.Tensor,
                          new_token_ids: list, base_tokenizer_path: str,
                          ext_tokenizer_path: str):
    """
    Compute per-token metrics for new tokens.

    For each new token:
      - pred = init_embs[token_id]  (the initialized embedding)
      - Compare against: the subword decomposition's mean embedding (what mean init should approximate)
      - Also compute norm stats

    Since there is no "gold" embedding for a new token (it doesn't exist in the base vocab),
    we measure:
      1. Norm statistics (pred_norm, ratio to base mean norm)
      2. Cosine distance to each subword's embedding (how well it represents its parts)
      3. Logit-space rank: where the new token ranks when we dot-product its embedding
         with all base embeddings (a good init should NOT rank highly — it should be orthogonal/uniform)
    """
    from transformers import AutoTokenizer

    base_tok = AutoTokenizer.from_pretrained(base_tokenizer_path)
    ext_tok = AutoTokenizer.from_pretrained(ext_tokenizer_path)

    V_base = base_embs.shape[0]
    base_norms = base_embs.norm(dim=1)
    base_mean_norm = base_norms.mean().item()

    results = {}

    for tid in tqdm(new_token_ids):
        pred = init_embs[tid].float()  # [H]

        # Decode token text
        decoded = ext_tok.decode([tid])

        # Encode with base tokenizer to get subword IDs
        subword_ids = base_tok.encode(decoded, add_special_tokens=False)
        if not subword_ids:
            subword_ids = [0]

        # Subword embeddings
        subword_embs = base_embs[subword_ids].float()  # [K, H]
        subword_mean = subword_embs.mean(dim=0)  # [H]

        # Cosine distance to subword mean
        cos_to_mean = 1.0 - F.cosine_similarity(pred.unsqueeze(0), subword_mean.unsqueeze(0)).item()

        # Centered cosine (remove anisotropy)
        base_mean_vec = base_embs.float().mean(dim=0)
        pred_c = pred - base_mean_vec
        subword_mean_c = subword_mean - base_mean_vec
        centered_cos = 1.0 - F.cosine_similarity(pred_c.unsqueeze(0), subword_mean_c.unsqueeze(0)).item()

        # Norm stats
        pred_norm = pred.norm().item()
        norm_ratio = pred_norm / base_mean_norm

        # MSE to subword mean
        mse_to_mean = F.mse_loss(pred, subword_mean).item()

        # Cosine distance to each individual subword
        cos_to_subwords = []
        for i, sid in enumerate(subword_ids):
            sw_emb = base_embs[sid].float()
            cos_d = 1.0 - F.cosine_similarity(pred.unsqueeze(0), sw_emb.unsqueeze(0)).item()
            cos_to_subwords.append(cos_d)

        results[str(tid)] = {
            "token_id": tid,
            "token_decoded": decoded,
            "num_base_fragments": len(subword_ids),
            "subword_ids": subword_ids,
            # Norm
            "pred_norm": pred_norm,
            "base_mean_norm": base_mean_norm,
            "norm_ratio": norm_ratio,
            # Direction
            "cos_to_subword_mean": cos_to_mean,
            "centered_cos_to_subword_mean": centered_cos,
            "mse_to_subword_mean": mse_to_mean,
            "cos_to_subwords": cos_to_subwords,
            "cos_to_subwords_mean": sum(cos_to_subwords) / len(cos_to_subwords),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate initialization quality (direct embedding comparison)")
    parser.add_argument("--init-model", type=str, required=True,
                        help="Path to assembled model (with mean/mlp/etc init)")
    parser.add_argument("--base-model", type=str, default="/workdir/models/Qwen3.5-2B-Base",
                        help="Path to base model (for gold subword embeddings)")
    parser.add_argument("--ext-tokenizer", type=str,
                        default="/workdir/models/RuadaptQwen3.5-2B-Base-b64-minlen4-mean-v2_tokenizer",
                        help="Path to extended tokenizer (to identify new tokens)")
    parser.add_argument("--passport", type=str, default=None,
                        help="Path to token_passport.json (optional)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    parser.add_argument("--num-tokens", type=int, default=None,
                        help="Evaluate only first N tokens (for quick testing)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find new token IDs
    print("Finding new token IDs...")
    new_token_ids = find_new_token_ids(args.base_model, args.ext_tokenizer)
    print(f"  Found {len(new_token_ids)} new tokens (IDs {new_token_ids[0]}..{new_token_ids[-1]})")

    if args.num_tokens:
        new_token_ids = new_token_ids[:args.num_tokens]
        print(f"  Evaluating first {len(new_token_ids)} tokens")

    # Load models
    from transformers import AutoModelForCausalLM

    print(f"Loading init model from {args.init_model}...")
    init_model = load_causal_lm(args.init_model, dtype=torch.float32)
    init_embs = init_model.get_input_embeddings().weight.detach().to(device)
    print(f"  Embeddings shape: {init_embs.shape}")
    del init_model

    print(f"Loading base model from {args.base_model}...")
    base_model = load_causal_lm(args.base_model, dtype=torch.float32)
    base_embs = base_model.get_input_embeddings().weight.detach().to(device)
    print(f"  Embeddings shape: {base_embs.shape}")
    del base_model

    # Compute metrics
    print("Computing per-token metrics...")
    results = compute_init_metrics(
        init_embs, base_embs, new_token_ids,
        args.base_model, args.ext_tokenizer
    )

    # Load passport if available
    if args.passport and Path(args.passport).exists():
        print(f"Loading passport from {args.passport}...")
        with open(args.passport) as f:
            passports = json.load(f)
        passport_map = {str(p["token_id"]): p for p in passports}
        for tid_str, stats in results.items():
            if tid_str in passport_map:
                pp = passport_map[tid_str]
                stats["token_type"] = pp.get("token_type")
                stats["bpe_depth"] = pp.get("bpe_depth")
                stats["corpus_frequency"] = pp.get("corpus_frequency")
                stats["corpus_bucket"] = pp.get("corpus_bucket")

    # Aggregate stats
    all_cos = [s["cos_to_subword_mean"] for s in results.values()]
    all_centered = [s["centered_cos_to_subword_mean"] for s in results.values()]
    all_mse = [s["mse_to_subword_mean"] for s in results.values()]
    all_norm_ratio = [s["norm_ratio"] for s in results.values()]
    all_cos_sw = [s["cos_to_subwords_mean"] for s in results.values()]

    agg = {
        "cos_to_subword_mean": sum(all_cos) / len(all_cos),
        "centered_cos_to_subword_mean": sum(all_centered) / len(all_centered),
        "mse_to_subword_mean": sum(all_mse) / len(all_mse),
        "norm_ratio_mean": sum(all_norm_ratio) / len(all_norm_ratio),
        "cos_to_subwords_mean": sum(all_cos_sw) / len(all_cos_sw),
        "num_tokens": len(results),
    }

    # Build output
    output = {
        "meta": {
            "init_model": args.init_model,
            "base_model": args.base_model,
            "ext_tokenizer": args.ext_tokenizer,
            "num_new_tokens": len(results),
            "aggregate_metrics": agg,
        },
        "tokens": results,
    }

    # Save
    if args.output is None:
        args.output = str(PROJECT_ROOT / "results" / "init_quality.json")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {args.output}")
    print(f"File size: {Path(args.output).stat().st_size / 1024 / 1024:.1f} MB")

    print("\n=== Aggregate Metrics ===")
    print(f"  cos_to_subword_mean:          {agg['cos_to_subword_mean']:.4f}")
    print(f"  centered_cos_to_subword_mean: {agg['centered_cos_to_subword_mean']:.4f}")
    print(f"  mse_to_subword_mean:          {agg['mse_to_subword_mean']:.6f}")
    print(f"  norm_ratio_mean:              {agg['norm_ratio_mean']:.4f}")
    print(f"  cos_to_subwords_mean:         {agg['cos_to_subwords_mean']:.4f}")


if __name__ == "__main__":
    main()
