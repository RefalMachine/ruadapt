"""
analyze_results.py — Load training logs and produce a Markdown comparison table.

Usage:
    python analyze_results.py --results-dir results/loss_sweep_v2
    python analyze_results.py --results-dir results/loss_sweep_v2 --sort-by logit_top1_acc
    python analyze_results.py --results-dir results/loss_sweep_v2 --csv results_table.csv
"""

import os
import json
import argparse


def load_all_logs(results_dir: str) -> dict[str, dict]:
    """Load all *_logs.json files from a directory."""
    logs = {}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith("_logs.json"):
            continue
        path = os.path.join(results_dir, fname)
        with open(path) as f:
            data = json.load(f)
        run_name = fname.replace("_logs.json", "")
        logs[run_name] = data
    return logs


def extract_last_metrics(logs: dict[str, dict]) -> tuple[list[str], list[str], list[list]]:
    """
    Extract the last value of each metric from each run.
    Returns: (run_names, metric_names, rows)
    """
    all_metric_names = set()
    for log in logs.values():
        all_metric_names.update(log.get("metrics", {}).keys())

    priority = [
        "loss", "logit_mrr", "logit_top1_acc", "logit_top5_acc", "logit_top10_acc",
        "centered_cos_dist", "cosine_dist", "norm_error", "mse",
        "soft_kl_0.1", "soft_kl_0.5", "soft_kl_1.0",
        "mrr", "reldist",
    ]
    metric_names = [m for m in priority if m in all_metric_names]
    remaining = sorted(all_metric_names - set(metric_names))
    metric_names.extend(remaining)

    run_names = list(logs.keys())
    rows = []

    for run_name in run_names:
        metrics = logs[run_name].get("metrics", {})
        row = []
        for m in metric_names:
            series = metrics.get(m, [])
            row.append(series[-1] if series else None)
        rows.append(row)

    return run_names, metric_names, rows


def format_value(val) -> str:
    if val is None:
        return "—"
    if abs(val) < 0.001:
        return f"{val:.9f}"
    if abs(val) < 1.0:
        return f"{val:.4f}"
    return f"{val:.2f}"


def format_markdown(
    run_names: list[str],
    metric_names: list[str],
    rows: list[list],
    sort_by: str = None,
    sort_dir: str = "max",
) -> str:
    """Format as Markdown table, optionally sorted."""
    if sort_by and sort_by in metric_names:
        col_idx = metric_names.index(sort_by)
        indices = list(range(len(run_names)))
        indices.sort(
            key=lambda i: (
                rows[i][col_idx]
                if rows[i][col_idx] is not None
                else float("-inf") if sort_dir == "max" else float("inf")
            ),
            reverse=(sort_dir == "max"),
        )
        run_names = [run_names[i] for i in indices]
        rows = [rows[i] for i in indices]

    lines = []
    lines.append("| Run | " + " | ".join(metric_names) + " |")
    lines.append("|:---| " + " | ".join("---:" for _ in metric_names) + " |")
    for i, name in enumerate(run_names):
        vals = " | ".join(format_value(rows[i][j]) for j in range(len(metric_names)))
        lines.append(f"| {name} | {vals} |")

    return "\n".join(lines)


def to_csv(run_names, metric_names, rows, path):
    with open(path, "w") as f:
        f.write("run," + ",".join(metric_names) + "\n")
        for i, name in enumerate(run_names):
            vals = ",".join(
                f"{rows[i][j]:.6f}" if rows[i][j] is not None else ""
                for j in range(len(metric_names))
            )
            f.write(f"{name},{vals}\n")
    print(f"Saved CSV → {path}")


def main():
    parser = argparse.ArgumentParser(description="Compare training runs (last eval point).")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--sort-by", type=str, default="logit_mrr")
    parser.add_argument("--sort-dir", type=str, default="max", choices=["max", "min"])
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    logs = load_all_logs(args.results_dir)
    if not logs:
        print(f"No *_logs.json files found in {args.results_dir}")
        return

    print(f"Found {len(logs)} runs in {args.results_dir}\n")

    run_names, metric_names, rows = extract_last_metrics(logs)

    table = format_markdown(
        run_names, metric_names, rows,
        sort_by=args.sort_by,
        sort_dir=args.sort_dir,
    )
    print(table)

    if args.csv:
        to_csv(run_names, metric_names, rows, args.csv)


if __name__ == "__main__":
    main()