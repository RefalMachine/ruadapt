#!/usr/bin/env python3
"""
Stage 2c: Lifecycle Analysis Script

Join passport + per-token metrics, produce stratified analysis and visualizations.

Usage:
    python analytics/analyze_token_lifecycle.py \
        --per-token-metrics results/<run>/per_token_init_metrics.json \
        --passport data/token_passport.json \
        --output-dir results/<run>/lifecycle_analysis/
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def filter_special_tokens(tokens: Dict[str, Dict], tokenizer_path: str = None) -> Dict[str, Dict]:
    """Remove special/added tokens from the token dict."""
    if tokenizer_path is None:
        return tokens
    from transformers import AutoTokenizer
    from ruadapt.tokenization.utils import get_special_token_ids
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    special_ids = get_special_token_ids(tok)
    before = len(tokens)
    filtered = {tid: stats for tid, stats in tokens.items() if int(tid) not in special_ids}
    print(f"  Filtered {before - len(filtered)} special tokens, {len(filtered)} remain")
    return filtered


def load_per_token_metrics(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def load_passport(path: str) -> Dict[str, Dict]:
    with open(path) as f:
        passports = json.load(f)
    return {str(p["token_id"]): p for p in passports}


def stratified_summary(tokens: Dict[str, Dict], group_by: str) -> Dict:
    """Compute mean metrics grouped by a field."""
    groups = defaultdict(lambda: {"count": 0, "metrics": defaultdict(list)})

    for tid, stats in tokens.items():
        group_val = stats.get(group_by, "unknown")
        if group_val is None:
            group_val = "unknown"
        group_val = str(group_val)  # ensure string key for JSON

        g = groups[group_val]
        g["count"] += 1

        # Head-mode metrics (nested dicts with "mean" key)
        for metric in ["mse", "cos_dist", "centered_cos_dist", "norm_error"]:
            if metric in stats and isinstance(stats[metric], dict):
                g["metrics"][metric].append(stats[metric]["mean"])

        if "logit_rank" in stats and isinstance(stats["logit_rank"], dict):
            g["metrics"]["logit_rank_mean"].append(stats["logit_rank"]["mean"])
            g["metrics"]["logit_mrr"].append(1.0 / stats["logit_rank"]["mean"])

        if "rank_distribution" in stats:
            rd = stats["rank_distribution"]
            g["metrics"]["top1_pct"].append(rd.get("top1_pct", 0))
            g["metrics"]["gt100_pct"].append(rd.get("gt100_pct", 0))

        # Direct-mode metrics (scalar values)
        for metric in ["cos_to_subword_mean", "centered_cos_to_subword_mean",
                        "mse_to_subword_mean", "norm_ratio", "cos_to_subwords_mean"]:
            if metric in stats and isinstance(stats[metric], (int, float)):
                g["metrics"][metric].append(stats[metric])

    # Compute means
    result = {}
    for group_val, data in groups.items():
        result[group_val] = {"count": data["count"]}
        for metric, vals in data["metrics"].items():
            if vals:
                result[group_val][metric] = {
                    "mean": sum(vals) / len(vals),
                    "min": min(vals),
                    "max": max(vals),
                }
    return result


def compute_correlations(tokens: Dict[str, Dict]) -> Dict:
    """Compute Pearson correlations between token properties and metrics."""
    try:
        import numpy as np
    except ImportError:
        return {"error": "numpy not installed"}

    # Collect paired data — supports both head-mode and direct-mode metrics
    pairs = {
        "bpe_depth_vs_logit_rank": ("bpe_depth", "logit_rank_mean"),
        "bpe_depth_vs_cos_dist": ("bpe_depth", "cos_dist_mean"),
        "bpe_depth_vs_cos_to_sw_mean": ("bpe_depth", "cos_to_subword_mean"),
        "num_base_fragments_vs_logit_rank": ("num_base_fragments", "logit_rank_mean"),
        "num_base_fragments_vs_cos_dist": ("num_base_fragments", "cos_dist_mean"),
        "num_base_fragments_vs_cos_to_sw_mean": ("num_base_fragments", "cos_to_subword_mean"),
        "corpus_frequency_vs_logit_rank": ("corpus_frequency", "logit_rank_mean"),
        "corpus_frequency_vs_norm_error": ("corpus_frequency", "norm_error_mean"),
    }

    correlations = {}
    for name, (x_key, y_key) in pairs.items():
        xs, ys = [], []
        for tid, stats in tokens.items():
            x_val = stats.get(x_key)
            # Try nested dict first (head-mode), then scalar (direct-mode)
            if y_key == "logit_rank_mean":
                y_val = stats.get("logit_rank", {}).get("mean") if isinstance(stats.get("logit_rank"), dict) else None
            elif y_key == "cos_dist_mean":
                y_val = stats.get("cos_dist", {}).get("mean") if isinstance(stats.get("cos_dist"), dict) else None
            elif y_key == "norm_error_mean":
                y_val = stats.get("norm_error", {}).get("mean") if isinstance(stats.get("norm_error"), dict) else None
            elif y_key == "cos_to_subword_mean":
                y_val = stats.get("cos_to_subword_mean")  # scalar
            else:
                y_val = None

            if x_val is not None and y_val is not None:
                xs.append(float(x_val))
                ys.append(float(y_val))

        if len(xs) > 10:
            xs_arr = np.array(xs)
            ys_arr = np.array(ys)
            if xs_arr.std() > 0 and ys_arr.std() > 0:
                pearson = float(np.corrcoef(xs_arr, ys_arr)[0, 1])
            else:
                pearson = 0.0
            from scipy.stats import spearmanr
            spearman, _ = spearmanr(xs_arr, ys_arr)

            correlations[name] = {
                "pearson": pearson,
                "spearman": float(spearman),
                "n_samples": len(xs),
            }

    # Remove pairs with no data
    correlations = {k: v for k, v in correlations.items() if v.get("n_samples", 0) > 0}
    return correlations


def find_pathological(tokens: Dict[str, Dict], top_n: int = 50) -> Dict[str, List]:
    """Find worst tokens by various metrics. Supports both head-mode and direct-mode."""
    result = {}

    # Worst by logit_rank_mean (head-mode only — has logit_rank in per-token data)
    rank_list = []
    for tid, stats in tokens.items():
        if "logit_rank" in stats and isinstance(stats["logit_rank"], dict):
            rank_list.append((tid, stats["logit_rank"]["mean"], stats.get("token_decoded", "")))
    if rank_list:
        rank_list.sort(key=lambda x: -x[1])
        result["worst_logit_rank"] = [
            {"token_id": tid, "logit_rank_mean": r, "token_decoded": d}
            for tid, r, d in rank_list[:top_n]
        ]

    # Worst by norm_error (head-mode)
    norm_list = []
    for tid, stats in tokens.items():
        if "norm_error" in stats and isinstance(stats["norm_error"], dict):
            norm_list.append((tid, stats["norm_error"]["mean"], stats.get("token_decoded", "")))
    if norm_list:
        norm_list.sort(key=lambda x: -x[1])
        result["worst_norm_error"] = [
            {"token_id": tid, "norm_error_mean": n, "token_decoded": d}
            for tid, n, d in norm_list[:top_n]
        ]

    # Worst by cos_dist (head-mode)
    cos_list = []
    for tid, stats in tokens.items():
        if "cos_dist" in stats and isinstance(stats["cos_dist"], dict):
            cos_list.append((tid, stats["cos_dist"]["mean"], stats.get("token_decoded", "")))
    if cos_list:
        cos_list.sort(key=lambda x: -x[1])
        result["worst_cos_dist"] = [
            {"token_id": tid, "cos_dist_mean": c, "token_decoded": d}
            for tid, c, d in cos_list[:top_n]
        ]

    # Worst by gt100_pct (head-mode)
    gt100_list = []
    for tid, stats in tokens.items():
        if "rank_distribution" in stats:
            gt100 = stats["rank_distribution"].get("gt100_pct", 0)
            gt100_list.append((tid, gt100, stats.get("token_decoded", "")))
    if gt100_list:
        gt100_list.sort(key=lambda x: -x[1])
        result["worst_gt100_pct"] = [
            {"token_id": tid, "gt100_pct": p, "token_decoded": d}
            for tid, p, d in gt100_list[:top_n]
        ]

    # Worst by cos_to_subword_mean (direct-mode)
    cos_sw_list = []
    for tid, stats in tokens.items():
        if "cos_to_subword_mean" in stats and isinstance(stats["cos_to_subword_mean"], (int, float)):
            cos_sw_list.append((tid, stats["cos_to_subword_mean"], stats.get("token_decoded", "")))
    if cos_sw_list:
        cos_sw_list.sort(key=lambda x: -x[1])
        result["worst_cos_to_subword_mean"] = [
            {"token_id": tid, "cos_to_subword_mean": c, "token_decoded": d}
            for tid, c, d in cos_sw_list[:top_n]
        ]

    # Worst by norm_ratio deviation from 1.0 (direct-mode)
    norm_ratio_list = []
    for tid, stats in tokens.items():
        if "norm_ratio" in stats and isinstance(stats["norm_ratio"], (int, float)):
            norm_ratio_list.append((tid, abs(stats["norm_ratio"] - 1.0), stats.get("token_decoded", "")))
    if norm_ratio_list:
        norm_ratio_list.sort(key=lambda x: -x[1])
        result["worst_norm_ratio_deviation"] = [
            {"token_id": tid, "norm_ratio_deviation": d, "token_decoded": dec}
            for tid, d, dec in norm_ratio_list[:top_n]
        ]

    return result


def generate_markdown_report(
    meta: Dict,
    stratified: Dict,
    correlations: Dict,
    pathological: Dict,
) -> str:
    """Generate Markdown report."""
    lines = ["# Token Lifecycle Analysis Report\n"]

    # Meta
    lines.append("## Overview\n")
    lines.append(f"- Head: `{meta.get('head_path', 'N/A')}`")
    lines.append(f"- Pooling: {meta.get('pooling', 'N/A')}")
    lines.append(f"- Val examples: {meta.get('num_val_examples', 'N/A')}")
    lines.append(f"- Unique tokens: {meta.get('num_unique_tokens', 'N/A')}")
    lines.append("")

    agg = meta.get("aggregate_metrics", {})
    if agg:
        lines.append("### Aggregate Metrics\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for k, v in agg.items():
            if isinstance(v, float):
                lines.append(f"| {k} | {v:.6f} |")
            else:
                lines.append(f"| {k} | {v} |")
        lines.append("")

    # Detect mode and available metrics from aggregate
    agg = meta.get("aggregate_metrics", {})
    has_logit_mrr = "logit_mrr_mean" in agg
    # Check what keys are actually in stratified data
    sample_bucket = next(iter(stratified.get("corpus_bucket", {}).values()), {})
    has_init_quality = "cos_to_subword_mean" in sample_bucket
    has_per_token_direct = "cos_dist" in sample_bucket and not has_logit_mrr
    has_logit_data = "logit_mrr" in sample_bucket

    # Stratified by corpus_bucket
    if "corpus_bucket" in stratified:
        lines.append("## Stratification by Corpus Frequency Bucket\n")
        has_real_buckets = any(b in stratified["corpus_bucket"] for b in ["dead", "rare", "uncommon", "common"])
        if not has_real_buckets:
            lines.append("*No corpus frequency data available (pass `--freq-path` to `build_token_passport.py`).*\n")
        if has_init_quality:
            lines.append("| Bucket | Count | cos_to_sw_mean | centered_cos | norm_ratio | mse_to_sw_mean |")
            lines.append("|--------|-------|----------------|--------------|------------|----------------|")
            bucket_order = ["dead", "rare", "uncommon", "common"] if has_real_buckets else ["unknown"]
            for bucket in bucket_order:
                if bucket in stratified["corpus_bucket"]:
                    d = stratified["corpus_bucket"][bucket]
                    n = d["count"]
                    cos = d.get("cos_to_subword_mean", {}).get("mean", 0)
                    cc = d.get("centered_cos_to_subword_mean", {}).get("mean", 0)
                    nr = d.get("norm_ratio", {}).get("mean", 0)
                    mse = d.get("mse_to_subword_mean", {}).get("mean", 0)
                    lines.append(f"| {bucket} | {n} | {cos:.4f} | {cc:.4f} | {nr:.4f} | {mse:.6f} |")
        elif has_logit_data:
            lines.append("| Bucket | Count | logit_mrr | cos_dist | norm_error | top1% | gt100% |")
            lines.append("|--------|-------|-----------|----------|------------|-------|--------|")
            bucket_order = ["dead", "rare", "uncommon", "common"] if has_real_buckets else ["unknown"]
            for bucket in bucket_order:
                if bucket in stratified["corpus_bucket"]:
                    d = stratified["corpus_bucket"][bucket]
                    n = d["count"]
                    mrr = d.get("logit_mrr", {}).get("mean", 0)
                    cos = d.get("cos_dist", {}).get("mean", 0)
                    ne = d.get("norm_error", {}).get("mean", 0)
                    t1 = d.get("top1_pct", {}).get("mean", 0)
                    gt = d.get("gt100_pct", {}).get("mean", 0)
                    lines.append(f"| {bucket} | {n} | {mrr:.4f} | {cos:.4f} | {ne:.4f} | {t1:.1f} | {gt:.1f} |")
        else:
            lines.append("| Bucket | Count | cos_dist | centered_cos_dist | norm_error |")
            lines.append("|--------|-------|----------|-------------------|------------|")
            bucket_order = ["dead", "rare", "uncommon", "common"] if has_real_buckets else ["unknown"]
            for bucket in bucket_order:
                if bucket in stratified["corpus_bucket"]:
                    d = stratified["corpus_bucket"][bucket]
                    n = d["count"]
                    cos = d.get("cos_dist", {}).get("mean", 0)
                    cc = d.get("centered_cos_dist", {}).get("mean", 0)
                    ne = d.get("norm_error", {}).get("mean", 0)
                    lines.append(f"| {bucket} | {n} | {cos:.4f} | {cc:.4f} | {ne:.4f} |")
        lines.append("")

    # Stratified by token_type
    if "token_type" in stratified:
        lines.append("## Stratification by Token Type\n")
        lines.append("*All new tokens are `intermediate` by definition — they exist because of BPE merge rules.*\n")
        if has_init_quality:
            lines.append("| Type | Count | cos_to_sw_mean | centered_cos | norm_ratio |")
            lines.append("|------|-------|----------------|--------------|------------|")
            for ttype in ["leaf", "intermediate"]:
                if ttype in stratified["token_type"]:
                    d = stratified["token_type"][ttype]
                    n = d["count"]
                    cos = d.get("cos_to_subword_mean", {}).get("mean", 0)
                    cc = d.get("centered_cos_to_subword_mean", {}).get("mean", 0)
                    nr = d.get("norm_ratio", {}).get("mean", 0)
                    lines.append(f"| {ttype} | {n} | {cos:.4f} | {cc:.4f} | {nr:.4f} |")
        elif has_logit_data:
            lines.append("| Type | Count | logit_mrr | cos_dist | norm_error |")
            lines.append("|------|-------|-----------|----------|------------|")
            for ttype in ["leaf", "intermediate"]:
                if ttype in stratified["token_type"]:
                    d = stratified["token_type"][ttype]
                    n = d["count"]
                    mrr = d.get("logit_mrr", {}).get("mean", 0)
                    cos = d.get("cos_dist", {}).get("mean", 0)
                    ne = d.get("norm_error", {}).get("mean", 0)
                    lines.append(f"| {ttype} | {n} | {mrr:.4f} | {cos:.4f} | {ne:.4f} |")
        else:
            lines.append("| Type | Count | cos_dist | centered_cos_dist | norm_error |")
            lines.append("|------|-------|----------|-------------------|------------|")
            for ttype in ["leaf", "intermediate"]:
                if ttype in stratified["token_type"]:
                    d = stratified["token_type"][ttype]
                    n = d["count"]
                    cos = d.get("cos_dist", {}).get("mean", 0)
                    cc = d.get("centered_cos_dist", {}).get("mean", 0)
                    ne = d.get("norm_error", {}).get("mean", 0)
                    lines.append(f"| {ttype} | {n} | {cos:.4f} | {cc:.4f} | {ne:.4f} |")
        lines.append("")

    # Stratified by bpe_depth
    if "bpe_depth" in stratified:
        lines.append("## Stratification by BPE Depth\n")
        if has_init_quality:
            lines.append("| Depth | Count | cos_to_sw_mean | centered_cos | norm_ratio |")
            lines.append("|-------|-------|----------------|--------------|------------|")
            for depth in sorted(stratified["bpe_depth"].keys(), key=lambda x: int(x) if str(x).isdigit() else 999):
                d = stratified["bpe_depth"][depth]
                n = d["count"]
                cos = d.get("cos_to_subword_mean", {}).get("mean", 0)
                cc = d.get("centered_cos_to_subword_mean", {}).get("mean", 0)
                nr = d.get("norm_ratio", {}).get("mean", 0)
                lines.append(f"| {depth} | {n} | {cos:.4f} | {cc:.4f} | {nr:.4f} |")
        elif has_logit_data:
            lines.append("| Depth | Count | logit_mrr | cos_dist | norm_error |")
            lines.append("|-------|-------|-----------|----------|------------|")
            for depth in sorted(stratified["bpe_depth"].keys(), key=lambda x: int(x) if str(x).isdigit() else 999):
                d = stratified["bpe_depth"][depth]
                n = d["count"]
                mrr = d.get("logit_mrr", {}).get("mean", 0)
                cos = d.get("cos_dist", {}).get("mean", 0)
                ne = d.get("norm_error", {}).get("mean", 0)
                lines.append(f"| {depth} | {n} | {mrr:.4f} | {cos:.4f} | {ne:.4f} |")
        else:
            lines.append("| Depth | Count | cos_dist | centered_cos_dist | norm_error |")
            lines.append("|-------|-------|----------|-------------------|------------|")
            for depth in sorted(stratified["bpe_depth"].keys(), key=lambda x: int(x) if str(x).isdigit() else 999):
                d = stratified["bpe_depth"][depth]
                n = d["count"]
                cos = d.get("cos_dist", {}).get("mean", 0)
                cc = d.get("centered_cos_dist", {}).get("mean", 0)
                ne = d.get("norm_error", {}).get("mean", 0)
                lines.append(f"| {depth} | {n} | {cos:.4f} | {cc:.4f} | {ne:.4f} |")
        lines.append("")

    # Correlations
    if correlations and "error" not in correlations:
        lines.append("## Correlations\n")
        lines.append("| Pair | Pearson | Spearman | N |")
        lines.append("|------|---------|----------|---|")
        for name, vals in correlations.items():
            p = vals.get("pearson", 0)
            s = vals.get("spearman", 0)
            n = vals.get("n_samples", 0)
            lines.append(f"| {name} | {p:.4f} | {s:.4f} | {n} |")
        lines.append("")

    # Pathological tokens
    if "worst_logit_rank" in pathological:
        lines.append("## Top-20 Worst Tokens by Logit Rank\n")
        lines.append("| # | Token | Decoded | Logit Rank Mean |")
        lines.append("|---|-------|---------|----------------|")
        for i, entry in enumerate(pathological["worst_logit_rank"][:20]):
            lines.append(f"| {i+1} | {entry['token_id']} | {entry['token_decoded']} | {entry['logit_rank_mean']:.1f} |")
        lines.append("")

    if "worst_norm_error" in pathological:
        lines.append("## Top-20 Worst Tokens by Norm Error\n")
        lines.append("| # | Token | Decoded | Norm Error Mean |")
        lines.append("|---|-------|---------|-----------------|")
        for i, entry in enumerate(pathological["worst_norm_error"][:20]):
            lines.append(f"| {i+1} | {entry['token_id']} | {entry['token_decoded']} | {entry['norm_error_mean']:.4f} |")
        lines.append("")

    if "worst_cos_to_subword_mean" in pathological:
        lines.append("## Top-20 Worst Tokens by Cosine to Subword Mean\n")
        lines.append("| # | Token | Decoded | cos_to_sw_mean |")
        lines.append("|---|-------|---------|----------------|")
        for i, entry in enumerate(pathological["worst_cos_to_subword_mean"][:20]):
            lines.append(f"| {i+1} | {entry['token_id']} | {entry['token_decoded']} | {entry['cos_to_subword_mean']:.4f} |")
        lines.append("")

    if "worst_norm_ratio_deviation" in pathological:
        lines.append("## Top-20 Worst Tokens by Norm Ratio Deviation\n")
        lines.append("| # | Token | Decoded | |norm_ratio - 1| |")
        lines.append("|---|-------|---------|------------------|")
        for i, entry in enumerate(pathological["worst_norm_ratio_deviation"][:20]):
            lines.append(f"| {i+1} | {entry['token_id']} | {entry['token_decoded']} | {entry['norm_ratio_deviation']:.4f} |")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Token lifecycle analysis")
    parser.add_argument("--per-token-metrics", type=str, required=True)
    parser.add_argument("--passport", type=str, default=None)
    parser.add_argument("--tokenizer-path", type=str, default=None,
                        help="Path to extended tokenizer (to filter out special tokens)")
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    # Load data
    print(f"Loading per-token metrics from {args.per_token_metrics}...")
    metrics_data = load_per_token_metrics(args.per_token_metrics)
    meta = metrics_data.get("meta", {})
    tokens = metrics_data.get("tokens", {})
    print(f"  Loaded {len(tokens)} tokens")

    # Load passport
    passport = {}
    if args.passport:
        print(f"Loading passport from {args.passport}...")
        passport = load_passport(args.passport)
        print(f"  Loaded {len(passport)} passport entries")

    # Merge passport data into tokens
    for tid_str, stats in tokens.items():
        if tid_str in passport:
            pp = passport[tid_str]
            stats["token_type"] = pp.get("token_type")
            stats["bpe_depth"] = pp.get("bpe_depth")
            stats["corpus_frequency"] = pp.get("corpus_frequency")
            stats["corpus_bucket"] = pp.get("corpus_bucket")
            stats["num_base_fragments"] = pp.get("num_base_fragments")
            stats["token_decoded"] = pp.get("token_decoded")

    # Filter out special tokens
    print("Filtering special tokens...")
    tokens = filter_special_tokens(tokens, args.tokenizer_path)

    # Output dir
    if args.output_dir is None:
        args.output_dir = str(PROJECT_ROOT / "results" / "lifecycle_analysis")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stratified summaries
    print("Computing stratified summaries...")
    stratified = {}
    for group_by in ["corpus_bucket", "token_type", "bpe_depth", "num_base_fragments"]:
        stratified[group_by] = stratified_summary(tokens, group_by)

    # Correlations
    print("Computing correlations...")
    correlations = compute_correlations(tokens)

    # Pathological tokens
    print("Finding pathological tokens...")
    pathological = find_pathological(tokens)

    # Save JSON outputs
    print("Saving outputs...")

    with open(out_dir / "stratified_summary.json", "w") as f:
        json.dump(stratified, f, indent=2)

    with open(out_dir / "correlation_matrix.json", "w") as f:
        json.dump(correlations, f, indent=2)

    with open(out_dir / "pathological_tokens.json", "w") as f:
        json.dump(pathological, f, indent=2, ensure_ascii=False)

    # Generate report
    report = generate_markdown_report(meta, stratified, correlations, pathological)
    with open(out_dir / "lifecycle_report.md", "w") as f:
        f.write(report)

    print(f"\nSaved analysis to {out_dir}/")
    print(f"  - lifecycle_report.md")
    print(f"  - stratified_summary.json")
    print(f"  - correlation_matrix.json")
    print(f"  - pathological_tokens.json")


if __name__ == "__main__":
    main()
