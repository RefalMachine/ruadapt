"""Dataset statistics: collection, printing, histogram.

Usage:
    from ruadapt.training.datasets.stats import print_dataset_stats, plot_fragment_histogram, save_dataset_stats

    # After dataset creation:
    print_dataset_stats(dataset, tokenizer)
    plot_fragment_histogram(dataset.stats, save_path="stats/frag_hist.png")
    save_dataset_stats(dataset.stats, "stats/dataset_stats.json")
"""

import json
import os
from collections import Counter
from typing import Any, Dict, Optional


def print_dataset_stats(dataset: Any, tokenizer: Any, top_k: int = 20) -> None:
    """Print dataset statistics to stdout.

    Args:
        dataset: Dataset with .stats dict (PackedDatasetWithTargetSubstitution).
        tokenizer: HF tokenizer for decoding token IDs.
        top_k: Number of top tokens to display.
    """
    stats = dataset.stats if hasattr(dataset, "stats") else {}
    if not stats:
        print("[stats] No stats available on this dataset.")
        return

    total = stats.get("total_tokens", 0)
    usable = stats.get("usable_tokens", 0)
    frag = stats.get("fragmented_tokens", 0)
    subs = stats.get("target_substitutions", 0)
    frag_ratio = stats.get("fragment_ratio_actual", 0.0)
    frag_len_dist = stats.get("fragment_length_distribution", {})
    n_chunks = stats.get("n_chunks", 0)
    max_len = stats.get("max_length", 0)
    n_docs = stats.get("n_documents", 0)
    cfg_ratio = stats.get("fragment_ratio", 0.0)
    cfg_psplit = stats.get("p_split", 0.0)
    freeze_idx = stats.get("freeze_idx")
    total_vocab = stats.get("total_vocab_size", 0)
    trainable_vocab = stats.get("trainable_vocab_size", 0)

    print()
    print("=" * 60)
    print("  DATASET STATISTICS")
    print("=" * 60)
    print(f"  Documents:            {n_docs:,}")
    print(f"  Chunks:               {n_chunks:,} x {max_len}")
    print(f"  Total tokens:         {total:,}")
    print(f"  Usable tokens:        {usable:,}")
    if freeze_idx is not None:
        print(f"  Freeze index:         {freeze_idx:,}")
    if total_vocab:
        print(f"  Total vocab:          {total_vocab:,}")
    if trainable_vocab:
        print(f"  Trainable vocab:      {trainable_vocab:,}")
    print(f"  Fragment ratio (cfg): {cfg_ratio:.1%}")
    print(f"  p_split (cfg):        {cfg_psplit}")
    print(f"  Fragmented tokens:    {frag:,} ({frag_ratio:.1%} of total)")
    print(f"  Target substitutions: {subs:,}")

    # Fragment length distribution
    if frag_len_dist:
        total_frags = sum(frag_len_dist.values())
        print()
        print("  Fragment length distribution:")
        for flen in sorted(frag_len_dist.keys()):
            count = frag_len_dist[flen]
            pct = 100 * count / max(total_frags, 1)
            bar = "#" * int(pct / 2)
            print(f"    len={flen}:  {count:>8,}  ({pct:5.1f}%)  {bar}")

    # Top-K tokens
    token_freq = stats.get("token_frequency")
    if token_freq and isinstance(token_freq, Counter):
        print()
        print(f"  Top-{top_k} tokens (by frequency in input_ids):")
        print(f"    {'Rank':>4}  {'Token ID':>8}  {'Count':>10}  {'Decoded'}")
        print(f"    {'----':>4}  {'--------':>8}  {'-----':>10}  {'-------'}")
        for rank, (tok_id, count) in enumerate(
            token_freq.most_common(top_k), start=1
        ):
            try:
                decoded = tokenizer.decode([tok_id], skip_special_tokens=False)
                decoded_repr = repr(decoded) if len(decoded) > 30 else decoded
                if len(decoded_repr) > 40:
                    decoded_repr = decoded_repr[:37] + "..."
            except Exception:
                decoded_repr = "<?>"
            print(f"    {rank:>4}  {tok_id:>8}  {count:>10,}  {decoded_repr}")

    print("=" * 60)
    print()


def plot_fragment_histogram(
    stats: Dict[str, Any],
    save_path: Optional[str] = None,
    title: str = "Fragment Length Distribution",
) -> Optional[str]:
    """Plot histogram of fragment lengths and save to PNG.

    Args:
        stats: Dataset stats dict (with fragment_length_distribution).
        save_path: Path to save PNG. If None, auto-generates.
        title: Plot title.

    Returns:
        Path to saved PNG, or None if matplotlib unavailable.
    """
    frag_len_dist = stats.get("fragment_length_distribution", {})
    if not frag_len_dist:
        print("[stats] No fragment length data to plot.")
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[stats] matplotlib not available, skipping histogram plot.")
        return None

    lengths = sorted(frag_len_dist.keys())
    counts = [frag_len_dist[l] for l in lengths]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(
        [str(l) for l in lengths],
        counts,
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.5,
    )

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Fragment length (number of sub-tokens)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path is None:
        save_path = "fragment_length_histogram.png"

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[stats] Fragment histogram saved to {save_path}")
    return save_path


def plot_token_frequency_histogram(
    stats: Dict[str, Any],
    tokenizer: Any = None,
    save_path: Optional[str] = None,
    title: str = "Token Frequency Distribution",
) -> Optional[str]:
    """Plot histogram of token frequencies with auto-computed log-scale bins.

    First bin is always "freq=0" (tokens not seen in dataset).
    Remaining bins are log-spaced from 1 to max_freq.

    Args:
        stats: Dataset stats dict (with token_frequency Counter).
        tokenizer: HF tokenizer (unused, kept for API consistency).
        save_path: Path to save PNG.
        title: Plot title.

    Returns:
        Path to saved PNG, or None if matplotlib unavailable.
    """
    import numpy as np

    token_freq = stats.get("token_frequency")
    if not token_freq or not isinstance(token_freq, Counter):
        print("[stats] No token frequency data to plot.")
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[stats] matplotlib not available, skipping histogram plot.")
        return None

    # Total vocab size from stats (or estimate)
    total_vocab = stats.get("total_vocab_size", 0) or 291584
    freeze_idx = stats.get("freeze_idx")
    seen_token_ids = set(token_freq.keys())

    # When freeze_idx is set, compute n_zero only for non-special trainable range
    # Special tokens (id > eos_id) are excluded — they're never in training data
    if freeze_idx is not None and tokenizer is not None:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None and eos_id > freeze_idx:
            trainable_range_size = eos_id - freeze_idx
        else:
            trainable_range_size = total_vocab - freeze_idx
        seen_in_range = sum(1 for tid in seen_token_ids if freeze_idx <= tid < (eos_id or total_vocab))
        n_zero = max(0, trainable_range_size - seen_in_range)
    elif freeze_idx is not None:
        trainable_range_size = total_vocab - freeze_idx
        seen_in_range = sum(1 for tid in seen_token_ids if tid >= freeze_idx)
        n_zero = max(0, trainable_range_size - seen_in_range)
    else:
        n_zero = max(0, total_vocab - len(seen_token_ids))

    # Frequencies > 0
    nonzero_freqs = np.array([c for c in token_freq.values() if c > 0])
    if len(nonzero_freqs) == 0:
        print("[stats] No tokens with frequency > 0.")
        return None

    max_freq = int(nonzero_freqs.max())

    # Auto-compute log-scale bins from 1 to max_freq
    # Number of bins: ~15-20 for good resolution
    n_bins = min(20, max(5, int(np.log10(max_freq)) * 5))
    log_edges = np.logspace(0, np.log10(max_freq), n_bins + 1)
    log_edges = np.unique(np.round(log_edges).astype(int))  # deduplicate
    if log_edges[0] < 1:
        log_edges[0] = 1

    # Histogram for nonzero frequencies
    hist, edges = np.histogram(nonzero_freqs, bins=log_edges)

    # Build labels and counts: [0] + auto bins
    bin_labels = ["0"]
    bin_counts = [n_zero]

    for i in range(len(hist)):
        lo, hi = int(edges[i]), int(edges[i + 1])
        if lo == hi:
            bin_labels.append(str(lo))
        else:
            bin_labels.append(f"{lo}-{hi}")
        bin_counts.append(int(hist[i]))

    fig, ax = plt.subplots(figsize=(max(10, len(bin_labels) * 0.6), 5))
    x_pos = range(len(bin_labels))
    bars = ax.bar(
        x_pos,
        bin_counts,
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.5,
    )

    for bar, count in zip(bars, bin_counts):
        if count > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(bin_counts) * 0.01,
                f"{count:,}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=45,
            )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Token frequency (occurrences in dataset)")
    ax.set_ylabel("Number of tokens")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yscale("log")
    ax.set_ylabel("Number of tokens (log scale)")

    plt.tight_layout()

    if save_path is None:
        save_path = "token_frequency_histogram.png"

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[stats] Token frequency histogram saved to {save_path}")
    return save_path


def save_dataset_stats(stats: Dict[str, Any], path: str) -> None:
    """Save dataset stats to JSON.

    Args:
        stats: Dataset stats dict.
        path: Output JSON path.
    """
    # Counter objects are not JSON-serializable, convert to dict
    serializable = {}
    for k, v in stats.items():
        if isinstance(v, Counter):
            serializable[k] = dict(v.most_common())
        else:
            serializable[k] = v

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"[stats] Stats saved to {path}")
