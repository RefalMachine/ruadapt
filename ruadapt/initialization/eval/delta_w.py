#!/usr/bin/env python3
"""
Stage 3: Delta-W Diagnostic Decomposition

Compare embeddings before and after mini-CPT to diagnose what was wrong
with initialization. Decomposes weight change into radial (norm) and
tangential (direction) components.

Usage:
    python pipeline/training/eval_delta_w.py \
        --pre-model /workdir/models/Qwen3.5-2B-Base_Eval_cos_norm \
        --post-model /workdir/models/Qwen3.5-2B-Base_Eval_cos_norm_calibrated_hf \
        --output results/delta_w/cos_norm_delta_w.json

    # Batch (all pairs):
    python pipeline/training/eval_delta_w.py --batch \
        --output-dir results/delta_w/
"""

import os
import argparse
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from ruadapt.utils.model_utils import load_causal_lm


def find_new_token_range(base_model, ext_model):
    """Determine new token IDs (present in ext but not in base)."""
    base_vocab_size = base_model.get_input_embeddings().weight.shape[0]
    ext_vocab_size = ext_model.get_input_embeddings().weight.shape[0]
    new_ids = list(range(base_vocab_size, ext_vocab_size))
    return new_ids, base_vocab_size


def compute_gold_directions(new_token_ids, ext_tokenizer, base_tokenizer, base_embeddings):
    """
    For each new token, compute the mean embedding of its base-tokenizer subwords.
    This is the "ideal" direction that mean init approximates.

    Steps: decode new token ID with ext_tokenizer -> encode text with base_tokenizer -> mean of subword embeddings.

    Returns: [N, H] tensor of normalized subword-mean directions.
    """
    base_vocab_size = base_embeddings.shape[0]

    # Decode new token IDs using the extended tokenizer
    decoded_texts = [ext_tokenizer.decode([tid]) for tid in new_token_ids]

    # Encode with base tokenizer to get subword decomposition
    encodings = base_tokenizer(
        decoded_texts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encodings["input_ids"]                    # [N, max_len] CPU
    attention_mask = encodings["attention_mask"].float()   # [N, max_len] CPU

    # Clamp to valid base vocab range (handle potential OOV)
    input_ids = input_ids.clamp(0, base_vocab_size - 1)

    # Look up subword embeddings (all on CPU)
    subword_embs = base_embeddings[input_ids]             # [N, max_len, H]
    mask = attention_mask.unsqueeze(-1)                    # [N, max_len, 1]

    # Masked mean
    sum_embs = (subword_embs * mask).sum(dim=1)           # [N, H]
    count = mask.sum(dim=1).clamp(min=1)                  # [N, 1]
    mean_embs = sum_embs / count                          # [N, H]

    # Normalize to get direction
    gold_dirs = F.normalize(mean_embs, p=2, dim=1)        # [N, H]
    return gold_dirs


@torch.no_grad()
def compute_delta_w_metrics(
    pre_embs: torch.Tensor,
    post_embs: torch.Tensor,
    new_token_ids: list,
    base_mean_emb: torch.Tensor,
    gold_dirs: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    Compute per-token Delta-W diagnostic metrics.

    Args:
        pre_embs: pre-CPT embeddings [V, H]
        post_embs: post-CPT embeddings [V, H]
        new_token_ids: list of new token IDs
        base_mean_emb: mean embedding of base vocab [H] (for anisotropy)
        gold_dirs: normalized subword-mean directions [N, H]
        device: computation device

    Returns:
        dict mapping token_id -> per-token metrics
    """
    pre = pre_embs[new_token_ids].float().to(device)    # [N, H]
    post = post_embs[new_token_ids].float().to(device)  # [N, H]
    delta = post - pre                                   # [N, H]
    base_mean = base_mean_emb.float().to(device)         # [H]

    # Original direction
    pre_norms = pre.norm(dim=1)                          # [N]
    pre_dir = F.normalize(pre, p=2, dim=1)               # [N, H]

    # Post direction
    post_norms = post.norm(dim=1)                        # [N]
    post_dir = F.normalize(post, p=2, dim=1)             # [N, H]

    # Delta decomposition
    delta_norms = delta.norm(dim=1)                      # [N]

    # Radial component: projection of delta onto original direction
    radial_magnitude = (delta * pre_dir).sum(dim=1)      # [N] (signed)
    radial = radial_magnitude.unsqueeze(1) * pre_dir     # [N, H]
    radial_norms = radial.norm(dim=1)                    # [N]

    # Tangential component: everything else
    tangential = delta - radial                           # [N, H]
    tangential_norms = tangential.norm(dim=1)             # [N]

    # ── Per-token metrics ──
    eps = 1e-9

    # 1. convergence_ratio: how much changed relative to init
    convergence_ratio = delta_norms / (pre_norms + eps)

    # 2. radial_fraction: what fraction of change was norm vs direction
    radial_fraction = radial_norms / (delta_norms + eps)

    # 3. norm_direction: +1 if norm increased, -1 if decreased
    norm_direction = torch.sign(radial_magnitude)

    # 4. direction_error: angular distance between pre and post directions
    cos_sim = (pre_dir * post_dir).sum(dim=1).clamp(-1, 1)
    direction_error = torch.acos(cos_sim)  # radians

    # 5. radial_change: norm ratio after/before
    radial_change = post_norms / (pre_norms + eps)

    # 6. tangential_change: absolute direction rotation magnitude
    tangential_change = tangential_norms

    # 7. anisotropy_shift: movement toward/away from embedding centroid
    anisotropy_shift = (delta * base_mean).sum(dim=1)

    # 8. gold_alignment: did CPT move toward subword-mean direction?
    gold_alignment = (F.normalize(delta, p=2, dim=1) * gold_dirs.to(device)).sum(dim=1)
    # Only meaningful when delta is non-trivial
    gold_alignment = torch.where(
        delta_norms > 1e-6, gold_alignment, torch.zeros_like(gold_alignment)
    )

    # Build results
    results = {}
    for i, tid in enumerate(new_token_ids):
        results[str(tid)] = {
            "convergence_ratio": convergence_ratio[i].item(),
            "radial_fraction": radial_fraction[i].item(),
            "norm_direction": norm_direction[i].item(),
            "direction_error": direction_error[i].item(),
            "radial_change": radial_change[i].item(),
            "tangential_change": tangential_change[i].item(),
            "anisotropy_shift": anisotropy_shift[i].item(),
            "gold_alignment": gold_alignment[i].item(),
            "delta_norm": delta_norms[i].item(),
            "pre_norm": pre_norms[i].item(),
            "post_norm": post_norms[i].item(),
        }

    return results


def compute_aggregate_stats(results: dict) -> dict:
    """Compute aggregate statistics over all tokens."""
    if not results:
        return {}

    keys = [
        "convergence_ratio", "radial_fraction", "norm_direction",
        "direction_error", "radial_change", "tangential_change",
        "anisotropy_shift", "gold_alignment", "delta_norm",
        "pre_norm", "post_norm",
    ]

    agg = {}
    for key in keys:
        vals = [r[key] for r in results.values() if key in r]
        if not vals:
            continue
        t = torch.tensor(vals)
        agg[key] = {
            "mean": t.mean().item(),
            "std": t.std().item() if len(t) > 1 else 0.0,
            "median": t.median().item(),
            "min": t.min().item(),
            "max": t.max().item(),
            "p25": t.quantile(0.25).item(),
            "p75": t.quantile(0.75).item(),
        }

    # Pattern detection
    radial_fracs = [r["radial_fraction"] for r in results.values()]
    norm_dirs = [r["norm_direction"] for r in results.values()]

    n = len(radial_fracs)
    if n > 0:
        pct_norm_fix = sum(1 for rf in radial_fracs if rf > 0.7) / n * 100
        pct_dir_fix = sum(1 for rf in radial_fracs if rf < 0.3) / n * 100
        pct_norm_up = sum(1 for nd in norm_dirs if nd > 0) / n * 100
        pct_norm_down = sum(1 for nd in norm_dirs if nd < 0) / n * 100

        agg["pattern_summary"] = {
            "pct_norm_dominated": pct_norm_fix,    # radial_fraction > 0.7
            "pct_direction_dominated": pct_dir_fix, # radial_fraction < 0.3
            "pct_norm_increased": pct_norm_up,      # norm_direction = +1
            "pct_norm_decreased": pct_norm_down,    # norm_direction = -1
        }

    return agg


def process_model_pair(
    pre_model_path: str,
    post_model_path: str,
    ext_tokenizer_path: str,
    base_model_path: str,
    passport_map: dict,
    device: torch.device,
) -> dict:
    """Process a single pre/post-CPT model pair."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pair_name = Path(pre_model_path).name
    print(f"\n{'='*60}")
    print(f"  {pair_name}")
    print(f"  Pre:  {pre_model_path}")
    print(f"  Post: {post_model_path}")
    print(f"{'='*60}")

    # Load models
    print("  Loading pre-CPT model...")
    pre_model = load_causal_lm(pre_model_path, dtype=torch.float32)
    pre_embs = pre_model.get_input_embeddings().weight.detach()
    del pre_model

    print("  Loading post-CPT model...")
    post_model = load_causal_lm(post_model_path, dtype=torch.float32)
    post_embs = post_model.get_input_embeddings().weight.detach()
    del post_model

    assert pre_embs.shape == post_embs.shape, \
        f"Shape mismatch: {pre_embs.shape} vs {post_embs.shape}"

    print("  Loading base model for reference...")
    base_model = load_causal_lm(base_model_path, dtype=torch.float32)
    base_embs = base_model.get_input_embeddings().weight.detach()
    base_vocab_size = base_embs.shape[0]
    del base_model

    new_token_ids = list(range(base_vocab_size, pre_embs.shape[0]))
    print(f"  New tokens: {base_vocab_size}..{pre_embs.shape[0]-1} ({len(new_token_ids)} tokens)")

    # Base mean embedding (for anisotropy)
    base_mean_emb = base_embs.mean(dim=0)

    # Gold directions (subword mean)
    print("  Computing gold directions (subword means)...")
    ext_tok = AutoTokenizer.from_pretrained(ext_tokenizer_path)
    base_tok = AutoTokenizer.from_pretrained(base_model_path)
    gold_dirs = compute_gold_directions(new_token_ids, ext_tok, base_tok, base_embs)
    del ext_tok, base_tok

    # Compute delta-W metrics
    print("  Computing Delta-W metrics...")
    results = compute_delta_w_metrics(
        pre_embs, post_embs, new_token_ids,
        base_mean_emb, gold_dirs, device,
    )

    # Merge passport metadata
    for tid_str, metrics in results.items():
        tid = int(tid_str)
        if tid_str in passport_map:
            pp = passport_map[tid_str]
            metrics["token_decoded"] = pp.get("token_decoded", "")
            metrics["bpe_depth"] = pp.get("bpe_depth")
            metrics["corpus_frequency"] = pp.get("corpus_frequency")
            metrics["corpus_bucket"] = pp.get("corpus_bucket")
            metrics["token_type"] = pp.get("token_type")
            metrics["num_base_fragments"] = pp.get("num_base_fragments")

    # Aggregate stats
    agg = compute_aggregate_stats(results)

    ext_vocab_size = pre_embs.shape[0]

    # Cleanup GPU
    del pre_embs, post_embs, base_embs, base_mean_emb, gold_dirs
    torch.cuda.empty_cache()

    return {
        "meta": {
            "pre_model": pre_model_path,
            "post_model": post_model_path,
            "base_model": base_model_path,
            "base_vocab_size": base_vocab_size,
            "ext_vocab_size": ext_vocab_size,
            "num_new_tokens": len(new_token_ids),
            "num_analyzed": len(results),
            "pair_name": pair_name,
        },
        "aggregate": agg,
        "tokens": results,
    }


def discover_model_pairs(models_dir: str) -> list:
    """
    Auto-discover pre/post-CPT model pairs.

    Convention: pre = <name>, post = <name>_calibrated_hf
    Also handles: mean-v2-new / mean-v2-new_calibrated_hf
    """
    models_path = Path(models_dir)
    all_models = sorted([d.name for d in models_path.iterdir() if d.is_dir()])

    # Find all _calibrated_hf models
    calibrated = {m for m in all_models if m.endswith("_calibrated_hf")}

    pairs = []
    for cal in calibrated:
        pre_name = cal.replace("_calibrated_hf", "")
        pre_path = models_path / pre_name
        post_path = models_path / cal
        if pre_path.is_dir():
            pairs.append((str(pre_path), str(post_path)))
        else:
            # Try common variations
            for suffix in ["-new", "-v2-new"]:
                alt = pre_name + suffix
                alt_path = models_path / alt
                if alt_path.is_dir():
                    pairs.append((str(alt_path), str(post_path)))
                    break

    return sorted(pairs, key=lambda x: Path(x[0]).name)


def stratum_summary(results: dict, group_by: str) -> dict:
    """Group results by a field and compute per-group means."""
    groups = defaultdict(lambda: {"count": 0, "metrics": defaultdict(list)})

    metric_keys = [
        "convergence_ratio", "radial_fraction", "norm_direction",
        "direction_error", "radial_change", "tangential_change",
        "anisotropy_shift", "gold_alignment",
    ]

    for tid_str, metrics in results.items():
        gval = metrics.get(group_by)
        if gval is None:
            gval = "unknown"
        gval = str(gval)

        g = groups[gval]
        g["count"] += 1
        for key in metric_keys:
            if key in metrics:
                g["metrics"][key].append(metrics[key])

    output = {}
    for gval, data in groups.items():
        entry = {"count": data["count"]}
        for key, vals in data["metrics"].items():
            if vals:
                t = torch.tensor(vals)
                entry[key] = {
                    "mean": t.mean().item(),
                    "std": t.std().item() if len(t) > 1 else 0.0,
                    "median": t.median().item(),
                }
        output[gval] = entry
    return output


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Delta-W Diagnostic Decomposition")
    parser.add_argument("--pre-model", type=str, default=None,
                        help="Path to pre-CPT model")
    parser.add_argument("--post-model", type=str, default=None,
                        help="Path to post-CPT model")
    parser.add_argument("--base-model", type=str,
                        default="/workdir/models/Qwen3.5-2B-Base",
                        help="Path to base model (for gold directions)")
    parser.add_argument("--ext-tokenizer", type=str,
                        default="/workdir/models/RuadaptQwen3.5-2B-Base-b64-minlen4-mean-v2_tokenizer",
                        help="Path to extended tokenizer (to decode new token IDs)")
    parser.add_argument("--passport", type=str,
                        default="data/token_passport.json",
                        help="Path to token passport JSON")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (single pair mode)")
    parser.add_argument("--batch", action="store_true",
                        help="Batch mode: process all discovered model pairs")
    parser.add_argument("--models-dir", type=str,
                        default="/workdir/models",
                        help="Directory containing models (for batch mode)")
    parser.add_argument("--output-dir", type=str,
                        default="results/delta_w",
                        help="Output directory (for batch mode)")
    parser.add_argument("--pairs", type=str, nargs="*", default=None,
                        help="Explicit list of pre-model names (batch mode). "
                             "If omitted, auto-discovers all pairs.")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load passport
    passport_map = {}
    if args.passport and Path(args.passport).exists():
        print(f"Loading passport from {args.passport}...")
        with open(args.passport) as f:
            passports = json.load(f)
        passport_map = {str(p["token_id"]): p for p in passports}
        print(f"  Loaded {len(passport_map)} entries")

    if args.batch:
        # ── Batch mode ──
        pairs = discover_model_pairs(args.models_dir)
        if args.pairs:
            names = set(args.pairs)
            pairs = [(pre, post) for pre, post in pairs if Path(pre).name in names]

        print(f"Discovered {len(pairs)} model pairs")
        for i, (pre, post) in enumerate(pairs):
            print(f"  [{i+1}] {Path(pre).name}")

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        all_summaries = {}

        for pre_path, post_path in pairs:
            pair_name = Path(pre_path).name
            try:
                result = process_model_pair(
                    pre_path, post_path,
                    args.ext_tokenizer, args.base_model,
                    passport_map, device,
                )
            except Exception as e:
                print(f"  ERROR processing {pair_name}: {e}")
                continue

            # Save per-pair result
            pair_file = out_dir / f"{pair_name}_delta_w.json"
            with open(pair_file, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"  Saved: {pair_file}")

            # Collect summary
            all_summaries[pair_name] = {
                "pre_model": pre_path,
                "post_model": post_path,
                "aggregate": result["aggregate"],
                "stratified_by_corpus_bucket": stratum_summary(result["tokens"], "corpus_bucket"),
                "stratified_by_bpe_depth": stratum_summary(result["tokens"], "bpe_depth"),
            }

        # Save cross-method summary
        summary_file = out_dir / "cross_method_summary.json"
        with open(summary_file, "w") as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)
        print(f"\nSaved cross-method summary: {summary_file}")

        # Generate comparative report
        report = generate_cross_method_report(all_summaries)
        report_file = out_dir / "delta_w_report.md"
        with open(report_file, "w") as f:
            f.write(report)
        print(f"Saved report: {report_file}")

    else:
        # ── Single pair mode ──
        if not args.pre_model or not args.post_model:
            parser.error("--pre-model and --post-model required in single-pair mode")

        result = process_model_pair(
            args.pre_model, args.post_model,
            args.ext_tokenizer, args.base_model,
            passport_map, device,
        )

        # Add stratified analysis
        result["stratified_by_corpus_bucket"] = stratum_summary(result["tokens"], "corpus_bucket")
        result["stratified_by_bpe_depth"] = stratum_summary(result["tokens"], "bpe_depth")

        output_path = args.output
        if output_path is None:
            name = Path(args.pre_model).name
            output_path = str(PROJECT_ROOT / "results" / "delta_w" / f"{name}_delta_w.json")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\nSaved: {output_path}")
        print(f"File size: {Path(output_path).stat().st_size / 1024:.1f} KB")

        # Print quick summary
        agg = result["aggregate"]
        print("\n=== Delta-W Summary ===")
        for key in ["convergence_ratio", "radial_fraction", "direction_error",
                     "radial_change", "gold_alignment"]:
            if key in agg:
                print(f"  {key}: mean={agg[key]['mean']:.4f}, median={agg[key]['median']:.4f}")

        if "pattern_summary" in agg:
            ps = agg["pattern_summary"]
            print(f"\n  Pattern: norm-dominated={ps['pct_norm_dominated']:.1f}%, "
                  f"dir-dominated={ps['pct_direction_dominated']:.1f}%, "
                  f"norm↑={ps['pct_norm_increased']:.1f}%, "
                  f"norm↓={ps['pct_norm_decreased']:.1f}%")


def generate_cross_method_report(all_summaries: dict) -> str:
    """Generate a Markdown report comparing Delta-W across init methods."""
    lines = ["# Delta-W Diagnostic Report\n"]
    lines.append("Comparison of weight changes during mini-CPT across initialization methods.\n")
    lines.append("## Method Comparison\n")
    lines.append("| Method | Conv.Ratio | Rad.Frac | Dir.Error | Rad.Change | Gold.Align | Norm↑% | Norm↓% |")
    lines.append("|--------|-----------|----------|-----------|------------|------------|--------|--------|")

    for name, data in sorted(all_summaries.items()):
        agg = data.get("aggregate", {})
        cr = agg.get("convergence_ratio", {}).get("mean", 0)
        rf = agg.get("radial_fraction", {}).get("mean", 0)
        de = agg.get("direction_error", {}).get("mean", 0)
        rc = agg.get("radial_change", {}).get("mean", 0)
        ga = agg.get("gold_alignment", {}).get("mean", 0)
        ps = agg.get("pattern_summary", {})
        nu = ps.get("pct_norm_increased", 0)
        nd = ps.get("pct_norm_decreased", 0)
        lines.append(f"| {name} | {cr:.4f} | {rf:.3f} | {de:.4f} | {rc:.4f} | {ga:.3f} | {nu:.1f} | {nd:.1f} |")

    lines.append("")

    # Stratification by corpus bucket for each method
    lines.append("## Stratification by Corpus Bucket\n")
    for name, data in sorted(all_summaries.items()):
        strat = data.get("stratified_by_corpus_bucket", {})
        if not strat:
            continue
        lines.append(f"### {name}\n")
        lines.append("| Bucket | Count | Conv.Ratio | Rad.Frac | Dir.Error | Rad.Change |")
        lines.append("|--------|-------|-----------|----------|-----------|------------|")
        for bucket in ["dead", "rare", "uncommon", "common"]:
            if bucket in strat:
                d = strat[bucket]
                n = d["count"]
                cr = d.get("convergence_ratio", {}).get("mean", 0)
                rf = d.get("radial_fraction", {}).get("mean", 0)
                de = d.get("direction_error", {}).get("mean", 0)
                rc = d.get("radial_change", {}).get("mean", 0)
                lines.append(f"| {bucket} | {n} | {cr:.4f} | {rf:.3f} | {de:.4f} | {rc:.4f} |")
        lines.append("")

    # Stratification by BPE depth
    lines.append("## Stratification by BPE Depth\n")
    for name, data in sorted(all_summaries.items()):
        strat = data.get("stratified_by_bpe_depth", {})
        if not strat:
            continue
        lines.append(f"### {name}\n")
        lines.append("| Depth | Count | Conv.Ratio | Rad.Frac | Dir.Error | Rad.Change |")
        lines.append("|-------|-------|-----------|----------|-----------|------------|")
        for depth in sorted(strat.keys(), key=lambda x: int(x) if x.isdigit() else 999):
            d = strat[depth]
            n = d["count"]
            cr = d.get("convergence_ratio", {}).get("mean", 0)
            rf = d.get("radial_fraction", {}).get("mean", 0)
            de = d.get("direction_error", {}).get("mean", 0)
            rc = d.get("radial_change", {}).get("mean", 0)
            lines.append(f"| {depth} | {n} | {cr:.4f} | {rf:.3f} | {de:.4f} | {rc:.4f} |")
        lines.append("")

    # Key findings
    lines.append("## Key Findings\n")

    # Find method with lowest convergence ratio
    conv_ratios = {}
    for name, data in all_summaries.items():
        cr = data.get("aggregate", {}).get("convergence_ratio", {}).get("mean", float("inf"))
        conv_ratios[name] = cr
    if conv_ratios:
        best = min(conv_ratios, key=conv_ratios.get)
        lines.append(f"- **Least CPT correction needed:** `{best}` (convergence_ratio={conv_ratios[best]:.4f})")

    # Find method with most balanced radial fraction
    radial_fracs = {}
    for name, data in all_summaries.items():
        rf = data.get("aggregate", {}).get("radial_fraction", {}).get("mean", 0.5)
        radial_fracs[name] = rf
    if radial_fracs:
        closest_to_half = min(radial_fracs, key=lambda x: abs(radial_fracs[x] - 0.5))
        lines.append(f"- **Most balanced norm/direction correction:** `{closest_to_half}` "
                     f"(radial_fraction={radial_fracs[closest_to_half]:.3f})")

    # Norm explosion recovery detection
    for name, data in all_summaries.items():
        ps = data.get("aggregate", {}).get("pattern_summary", {})
        if ps.get("pct_norm_decreased", 0) > 50:
            lines.append(f"- **{name}:** >50% of tokens had norm DECREASED during CPT "
                         f"→ likely norm explosion in init")

    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
