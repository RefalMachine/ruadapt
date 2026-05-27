#!/usr/bin/env python3
"""
Unified Per-Token Evaluation — compares init embeddings against gold.

Two modes:
  head   — Run trained head on validation cache (requires --head-path + --cache-dir)
  direct — Compare embeddings from two models directly (requires --init-model + --gold-model)

Both modes produce identical output format: per-token metrics + aggregate stats.

Usage (head mode):
    python pipeline/training/eval_per_token.py \
        --head-path results/loss_sweep_v2/..._head_final.pt \
        --pooling attention \
        --cache-dir cache \
        --output results/<run>/per_token_init_metrics.json

Usage (direct mode):
    python pipeline/training/eval_per_token.py \
        --init-model /workdir/models/RuadaptQwen3.5-2B-Base-b64-minlen4-mean-v2-new \
        --gold-model /workdir/models/RuadaptQwen3.5-2B-Base-b64-minlen4-mean-v2-new_calibrated_hf \
        --output results/mean_init/per_token_init_metrics.json
"""

import os

_tmp_dir = "tmp"
os.makedirs(_tmp_dir, exist_ok=True)
os.environ["TMPDIR"] = _tmp_dir

import json
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from ruadapt.initialization.head.train import LayerAttentionHead, _load_chunk
from ruadapt.initialization.head.metrics import FastMetrics
from ruadapt.utils.model_utils import load_causal_lm


def load_head(head_path: str, pooling: str, device: torch.device) -> LayerAttentionHead:
    """Load trained head checkpoint."""
    ckpt = torch.load(head_path, map_location="cpu", weights_only=False)

    lw_key = "layer_weights"
    if lw_key in ckpt:
        num_layers = ckpt[lw_key].shape[0]
    else:
        raise KeyError(f"Cannot find 'layer_weights' in checkpoint keys: {list(ckpt.keys())}")

    mlp_w1_key = "mlp.0.weight"
    if mlp_w1_key in ckpt:
        hidden_size = ckpt[mlp_w1_key].shape[1]
    else:
        raise KeyError(f"Cannot find '{mlp_w1_key}' in checkpoint keys: {list(ckpt.keys())}")

    head = LayerAttentionHead(hidden_size, num_layers, pooling=pooling)
    head.load_state_dict(ckpt)
    head.to(device)
    head.eval()
    return head


def find_new_token_ids(base_model_path: str, ext_tokenizer_path: str = None, gold_embs_shape: int = None, init_embs_shape: int = None):
    """Determine new token IDs. Multiple strategies."""
    if init_embs_shape and gold_embs_shape:
        # Direct: new tokens are those in init but not in gold
        # Actually, both have same vocab size — new tokens are those added beyond base vocab
        # Use the difference between extended and base tokenizer
        pass

    if ext_tokenizer_path:
        from transformers import AutoTokenizer
        base_tok = AutoTokenizer.from_pretrained(base_model_path)
        ext_tok = AutoTokenizer.from_pretrained(ext_tokenizer_path)
        base_vocab = set(base_tok.get_vocab().keys())
        ext_vocab = set(ext_tok.get_vocab().keys())
        added = ext_vocab - base_vocab
        ext_full = ext_tok.get_vocab()
        return sorted([ext_full[tok] for tok in added])

    # Fallback: if we know base vocab size, new tokens are everything above it
    if gold_embs_shape and init_embs_shape and init_embs_shape > gold_embs_shape:
        return list(range(gold_embs_shape, init_embs_shape))

    raise ValueError("Cannot determine new token IDs. Provide --ext-tokenizer or both models.")


def accumulate_summary(per_token: dict) -> dict:
    """Compute per-token summary stats and aggregate metrics."""
    tokens_output = {}

    has_ranks = any("logit_ranks" in metrics and len(metrics["logit_ranks"]) > 0
                    for metrics in per_token.values())

    for tid, metrics in tqdm(per_token.items(), desc="Summarizing"):
        token_stats = {}
        for key in ["mse", "cos_dist", "centered_cos_dist", "norm_error", "pred_norms", "target_norms"]:
            vals = metrics.get(key, [])
            if not vals:
                continue
            t = torch.tensor(vals)
            token_stats[key] = {
                "mean": t.mean().item(),
                "std": t.std().item() if len(t) > 1 else 0.0,
                "min": t.min().item(),
                "max": t.max().item(),
                "median": t.median().item(),
            }

        if has_ranks and "logit_ranks" in metrics and len(metrics["logit_ranks"]) > 0:
            ranks = torch.tensor(metrics["logit_ranks"], dtype=torch.float)
            token_stats["logit_rank"] = {
                "mean": ranks.mean().item(),
                "std": ranks.std().item() if len(ranks) > 1 else 0.0,
                "min": ranks.min().item(),
                "max": ranks.max().item(),
                "median": ranks.median().item(),
            }
            total_ranks = len(ranks)
            token_stats["rank_distribution"] = {
                "top1_pct": (ranks == 1).sum().item() / total_ranks * 100,
                "top5_pct": (ranks <= 5).sum().item() / total_ranks * 100,
                "top10_pct": (ranks <= 10).sum().item() / total_ranks * 100,
                "gt100_pct": (ranks > 100).sum().item() / total_ranks * 100,
            }

        token_stats["num_val_examples"] = len(metrics.get("logit_ranks", metrics.get("mse", [])))
        tokens_output[str(tid)] = token_stats

    # Aggregate metrics
    all_mse, all_cos, all_centered, all_norm_err, all_ranks = [], [], [], [], []
    for tid_str, stats in tokens_output.items():
        n = stats["num_val_examples"]
        all_mse.extend([stats["mse"]["mean"]] * n)
        all_cos.extend([stats["cos_dist"]["mean"]] * n)
        all_centered.extend([stats["centered_cos_dist"]["mean"]] * n)
        all_norm_err.extend([stats["norm_error"]["mean"]] * n)
        if "logit_rank" in stats:
            all_ranks.extend([stats["logit_rank"]["mean"]] * n)

    agg = {
        "mse_mean": sum(all_mse) / len(all_mse) if all_mse else 0,
        "cos_dist_mean": sum(all_cos) / len(all_cos) if all_cos else 0,
        "centered_cos_dist_mean": sum(all_centered) / len(all_centered) if all_centered else 0,
        "norm_error_mean": sum(all_norm_err) / len(all_norm_err) if all_norm_err else 0,
    }
    if all_ranks:
        agg["logit_mrr_mean"] = sum(1.0/r for r in all_ranks) / len(all_ranks)

    return tokens_output, agg


def run_head_mode(args, device):
    """Run head on validation cache, accumulate per-token metrics."""
    print(f"Loading head from {args.head_path}...")
    head = load_head(args.head_path, args.pooling, device)

    gold_model_path = args.gold_model or args.model_path
    print(f"Loading gold embeddings from {gold_model_path}...")
    model = load_causal_lm(gold_model_path, dtype=torch.float32)
    gold_embeddings = model.get_input_embeddings().weight.detach()
    del model
    print(f"  Embeddings shape: {gold_embeddings.shape}")

    fast_metrics = FastMetrics(gold_embeddings)

    val_dir = Path(args.cache_dir) / "val"
    val_chunks = sorted(val_dir.glob("chunk_*.pt*"))
    print(f"Found {len(val_chunks)} val chunks")

    per_token = defaultdict(lambda: {
        "mse": [], "cos_dist": [], "centered_cos_dist": [],
        "norm_error": [], "logit_ranks": [],
        "pred_norms": [], "target_norms": [],
    })

    total_examples = 0

    for chunk_path in tqdm(val_chunks, desc="Chunks"):
        chunk = _load_chunk(str(chunk_path))

        hs = chunk["hidden_states"].to(device)
        mask = chunk["attention_mask"].to(device)
        tids = chunk["target_ids"]

        N = hs.shape[0]

        for i in range(0, N, args.batch_size):
            batch_hs = hs[i:i+args.batch_size]
            batch_mask = mask[i:i+args.batch_size]
            batch_tids = tids[i:i+args.batch_size].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = head(batch_hs, batch_mask)

            preds = preds.float()
            batch_metrics = fast_metrics.compute_per_token_metrics(preds, batch_tids)

            for j in range(len(batch_tids)):
                tid = batch_tids[j].item()
                for key in batch_metrics:
                    per_token[tid][key].append(batch_metrics[key][j].item())

            total_examples += len(batch_tids)

        del hs, mask
        torch.cuda.empty_cache()

    print(f"\nProcessed {total_examples} val examples, {len(per_token)} unique tokens")

    tokens_output, agg = accumulate_summary(per_token)

    return tokens_output, agg, {
        "mode": "head",
        "head_path": args.head_path,
        "gold_model": gold_model_path,
        "pooling": args.pooling,
        "num_val_examples": total_examples,
        "num_unique_tokens": len(tokens_output),
    }


def run_direct_mode(args, device):
    """Compare init embeddings against gold directly (no head, no cache)."""

    print(f"Loading init model from {args.init_model}...")
    init_model = load_causal_lm(args.init_model, dtype=torch.float32)
    init_embs = init_model.get_input_embeddings().weight.detach()
    del init_model
    print(f"  Init embeddings shape: {init_embs.shape}")

    print(f"Loading gold model from {args.gold_model}...")
    gold_model = load_causal_lm(args.gold_model, dtype=torch.float32)
    gold_embs = gold_model.get_input_embeddings().weight.detach()
    del gold_model
    print(f"  Gold embeddings shape: {gold_embs.shape}")

    assert init_embs.shape == gold_embs.shape, \
        f"Shape mismatch: init={init_embs.shape} vs gold={gold_embs.shape}"

    # Determine new token IDs
    base_vocab_size = gold_embs.shape[0]
    # Both models have same vocab — new tokens are those beyond base vocab
    # We need to figure out where base vocab ends
    # Strategy: use the fact that init model has new tokens appended
    # The gold model was CPT'd from the same base, so same vocab size
    # We need external info about which IDs are new
    if args.ext_tokenizer:
        from transformers import AutoTokenizer
        base_tok = AutoTokenizer.from_pretrained(args.gold_model)
        ext_tok = AutoTokenizer.from_pretrained(args.ext_tokenizer)
        base_vocab = set(base_tok.get_vocab().keys())
        ext_vocab = set(ext_tok.get_vocab().keys())
        added = ext_vocab - base_vocab
        ext_full = ext_tok.get_vocab()
        new_token_ids = sorted([ext_full[tok] for tok in added])
    else:
        # Fallback: assume base model has fewer embeddings
        # Load base model to get its vocab size
        base_model = load_causal_lm(args.base_model, dtype=torch.float32)
        base_vocab_size = base_model.get_input_embeddings().weight.shape[0]
        del base_model
        new_token_ids = list(range(base_vocab_size, init_embs.shape[0]))

    print(f"  New tokens: {new_token_ids[0]}..{new_token_ids[-1]} ({len(new_token_ids)})")

    # Move init embeddings to device (gold stays on CPU — FastMetrics expects this)
    init_embs = init_embs.to(device)

    # FastMetrics with gold embeddings on CPU (matches head mode behavior)
    fast_metrics = FastMetrics(gold_embs)

    # Direct mode: no logit ranks (comparing against full vocab is meaningless for new tokens)
    per_token = defaultdict(lambda: {
        "mse": [], "cos_dist": [], "centered_cos_dist": [],
        "norm_error": [],
        "pred_norms": [], "target_norms": [],
    })

    batch_size = args.batch_size
    for i in tqdm(range(0, len(new_token_ids), batch_size), desc="Batches"):
        batch_ids = new_token_ids[i:i+batch_size]
        batch_tids = torch.tensor(batch_ids, device=device)
        preds = init_embs[batch_tids].float()  # [B, H]

        batch_metrics = fast_metrics.compute_per_token_metrics(preds, batch_tids, compute_ranks=False)

        for j, tid in enumerate(batch_ids):
            for key in batch_metrics:
                per_token[tid][key].append(batch_metrics[key][j].item())

    print(f"  Computed metrics for {len(new_token_ids)} tokens")

    tokens_output, agg = accumulate_summary(per_token)

    # Cleanup
    del init_embs, gold_embs, fast_metrics
    torch.cuda.empty_cache()

    return tokens_output, agg, {
        "mode": "direct",
        "init_model": args.init_model,
        "gold_model": args.gold_model,
        "num_new_tokens": len(new_token_ids),
        "num_unique_tokens": len(tokens_output),
    }


def main():
    parser = argparse.ArgumentParser(description="Unified per-token evaluation (head or direct mode)")

    # Head mode options
    parser.add_argument("--head-path", type=str, default=None,
                        help="Path to trained head checkpoint (head mode)")
    parser.add_argument("--pooling", type=str, default="attention")
    parser.add_argument("--cache-dir", type=str, default="cache")
    parser.add_argument("--batch-size", type=int, default=512)

    # Direct mode options
    parser.add_argument("--init-model", type=str, default=None,
                        help="Path to init model (direct mode: mean, random, etc.)")
    parser.add_argument("--gold-model", type=str, default=None,
                        help="Path to gold/reference model (post-CPT). Used in both modes.")

    # Common options
    parser.add_argument("--model-path", type=str, default="/workdir/models/Qwen3.5-2B-Base",
                        help="[legacy] Base model path. Use --gold-model instead.")
    parser.add_argument("--base-model", type=str, default="/workdir/models/Qwen3.5-2B-Base",
                        help="Base model path (for finding new token IDs in direct mode)")
    parser.add_argument("--ext-tokenizer", type=str, default=None,
                        help="Extended tokenizer path (to identify new tokens)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--keep-val-in-ram", action="store_true")
    parser.add_argument("--passport-path", type=str, default=None)

    args = parser.parse_args()

    # Validate mode
    has_head = args.head_path is not None
    has_init = args.init_model is not None

    if has_head and has_init:
        parser.error("Choose one mode: --head-path OR --init-model, not both")
    if not has_head and not has_init:
        parser.error("Provide either --head-path (head mode) or --init-model (direct mode)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run appropriate mode
    if has_head:
        tokens_output, agg, meta = run_head_mode(args, device)
    else:
        tokens_output, agg, meta = run_direct_mode(args, device)

    # Load passport if available
    passport_data = {}
    if args.passport_path and Path(args.passport_path).exists():
        print(f"Loading passport from {args.passport_path}...")
        with open(args.passport_path) as f:
            passports = json.load(f)
        for p in passports:
            passport_data[str(p["token_id"])] = p

    # Merge passport data
    if passport_data:
        for tid_str, stats in tokens_output.items():
            if tid_str in passport_data:
                pp = passport_data[tid_str]
                stats["token_type"] = pp.get("token_type")
                stats["bpe_depth"] = pp.get("bpe_depth")
                stats["corpus_frequency"] = pp.get("corpus_frequency")
                stats["corpus_bucket"] = pp.get("corpus_bucket")
                stats["num_base_fragments"] = pp.get("num_base_fragments")
                stats["token_decoded"] = pp.get("token_decoded")

    # Decode tokens missing token_decoded
    tokenizer_source = args.gold_model or args.model_path
    missing = [tid_str for tid_str, s in tokens_output.items() if "token_decoded" not in s]
    if missing:
        from transformers import AutoTokenizer
        print(f"Decoding {len(missing)} tokens not in passport...")
        tok = AutoTokenizer.from_pretrained(tokenizer_source)
        for tid_str in missing:
            tokens_output[tid_str]["token_decoded"] = tok.decode([int(tid_str)])

    # Build output
    output = {
        "meta": {**meta, "aggregate_metrics": agg},
        "tokens": tokens_output,
    }

    # Save
    output_path = args.output
    if output_path is None:
        output_path = str(PROJECT_ROOT / "results" / "per_token_init_metrics.json")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved per-token metrics to {output_path}")
    print(f"File size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")

    print("\n=== Aggregate Metrics ===")
    for k, v in agg.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
