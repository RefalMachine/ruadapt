"""Evaluation runner — thin wrapper for llmtf_open."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Run model evaluation via llmtf_open")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--tasks", type=str, required=True, help="Comma-separated task names")
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per task")
    args = parser.parse_args()

    print(f"Model: {args.model_path}")
    print(f"Tasks: {args.tasks}")
    print(f"Output: {args.output_dir}")

    try:
        from llmtf_open.runner import run_evaluation

        tasks = [t.strip() for t in args.tasks.split(",")]
        run_evaluation(
            model_path=args.model_path,
            tasks=tasks,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )
    except ImportError:
        print("ERROR: llmtf_open submodule not available.")
        print("Initialize it with: git submodule update --init ruadapt/evaluation/llmtf_open")
        sys.exit(1)


if __name__ == "__main__":
    main()
