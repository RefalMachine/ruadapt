"""
20_train_from_cache_opt.py — Train pooling heads using precomputed hidden states.

Loads precomputed hidden state chunks produced by 18_precompute_hidden_states.py.
The LLM is NOT loaded — training is head-only, ~10x faster than full-model training.

Architecture:
  - layer_weights [num_stored_layers] — learnable, softmax-normalized
  - Pooling head (last / mean / attention / global_query)
  - MLP projection [hidden_size → hidden_size*2 → hidden_size]
  - Loss: configurable via --loss-type (see metrics.py for all options)

Async prefetch design:
  - A background thread pre-loads the NEXT chunk while the current one is being trained.
  - Within each chunk, a standard DataLoader with num_workers handles per-batch I/O.
  - This ensures GPU never waits for disk I/O.

Usage:
  python 20_train_from_cache_opt.py --pooling attention --loss-type mse --epochs 5 --all-splits
  python 20_train_from_cache_opt.py --pooling attention --loss-type soft_logit_kl --loss-kwargs '{"tau": 0.5}'
  python 20_train_from_cache_opt.py --pooling attention --loss-type geometric_aware --loss-kwargs '{"tau": 0.5, "gamma": 2.0}'

Key flags:
  --pooling        last | mean | attention | global_query
  --all-splits     Use WeightedRandomSampler (50/30/20 len distribution). Recommended.
  --epochs         Full passes over the train set
  --max-steps      Override epochs (useful for quick smoke tests)
  --lr             Peak learning rate for cosine schedule (default 1e-3)
  --batch-size     Mini-batch size within a chunk (default 256, head-only is fast)
  --loss-type      Loss function (see metrics.py compute_loss docstring)
  --loss-kwargs    JSON dict of extra kwargs for the loss function
  --cache-dir      Path to cache produced by 18_precompute_hidden_states.py
  --model-path     Path to base LLM (needed for target embeddings only — no forward pass)
"""

import os
import sys

# Force all temporary files (including PyTorch file_system shm) to be created in /workdir
_tmp_dir = "tmp"
os.makedirs(_tmp_dir, exist_ok=True)
os.environ["TMPDIR"] = _tmp_dir
os.environ["TEMP"] = _tmp_dir
os.environ["TMP"] = _tmp_dir

import json
import math
import argparse
import io
import time
import concurrent.futures
import multiprocessing as mp
import threading
import queue
import torch
import torch.multiprocessing

# Workaround for small /dev/shm (Docker default 64MB)
torch.multiprocessing.set_sharing_strategy('file_system')

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoModelForCausalLM, AutoConfig, get_cosine_schedule_with_warmup
from tqdm import tqdm
from ruadapt.initialization.head.metrics import FastMetrics
from ruadapt.utils.model_utils import load_causal_lm


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_chunk(path: str) -> dict:
    """Load a chunk file, auto-detecting zstd compression from extension."""
    if path.endswith(".zst"):
        try:
            import zstandard as zstd
        except ImportError:
            raise RuntimeError("Chunk is zstd-compressed but zstandard is not installed. "
                               "Run: pip install zstandard")
        dctx = zstd.ZstdDecompressor()
        with open(path, "rb") as f:
            raw = dctx.decompress(f.read())
        return torch.load(io.BytesIO(raw), weights_only=True)
    return torch.load(path, weights_only=True, map_location="cpu")


# ---------------------------------------------------------------------------
# Cache Dataset (single chunk in memory)
# ---------------------------------------------------------------------------

class CachedChunkDataset(Dataset):
    """Wraps a single pre-loaded chunk dict."""

    def __init__(self, chunk: dict, target_embeddings: torch.Tensor,
                 use_all_splits: bool = True):
        self.hs      = chunk["hidden_states"]    # float16 CPU
        self.mask    = chunk["attention_mask"]   # bool   CPU
        self.tids    = chunk["target_ids"]       # int32  CPU
        self.slens   = chunk["seq_lens"].long()  # CPU
        self.embs    = target_embeddings
        self.all_splits = use_all_splits

    def __len__(self):
        return self.hs.shape[0]

    def __getitem__(self, idx):
        return {
            "hs":   self.hs[idx],
            "mask": self.mask[idx],
            "tid":  self.tids[idx],
            "slen": self.slens[idx],
        }

    def get_sampler_weights(self, target_dist=(0.5, 0.3, 0.2)):
        lens = self.slens.tolist()
        b2, b3, b4 = [], [], []
        for i, l in enumerate(lens):
            if l == 2:   b2.append(i)
            elif l == 3: b3.append(i)
            else:        b4.append(i)

        weights = [0.0] * len(lens)

        def assign(indices, mass):
            if not indices:
                return
            w = mass / len(indices)
            for i in indices:
                weights[i] = w

        assign(b2, target_dist[0])
        assign(b3, target_dist[1])
        assign(b4, target_dist[2])
        return torch.tensor(weights, dtype=torch.double)


def make_loader(chunk: dict, target_embeddings: torch.Tensor,
                batch_size: int, use_all_splits: bool) -> DataLoader:
    ds = CachedChunkDataset(chunk, target_embeddings, use_all_splits)
    if use_all_splits:
        weights = ds.get_sampler_weights()
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=0, pin_memory=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=0, pin_memory=True)


# ---------------------------------------------------------------------------
# Worker function for ProcessPoolExecutor (must be top-level)
# ---------------------------------------------------------------------------
def _load_chunk_worker(path: str, idx: int):
    t_start = time.time()
    if not path.endswith(".zst"):
        t_end = time.time()
        return path, idx, t_end - t_start, False

    import zstandard as zstd
    dctx = zstd.ZstdDecompressor()

    tmp_dir = os.environ.get("TMPDIR", "tmp")
    tmp_path = os.path.join(tmp_dir, f"chunk_uncompressed_{idx}.pt")

    with open(path, "rb") as ifh, open(tmp_path, "wb") as ofh:
        dctx.copy_stream(ifh, ofh)

    t_end = time.time()
    return tmp_path, idx, t_end - t_start, True


# ---------------------------------------------------------------------------
# Async Chunk Prefetcher (Two-Stage Pipeline)
# ---------------------------------------------------------------------------

class ChunkPrefetcher:
    """Two-stage async prefetch: zstd decompress (process) → torch.load (thread) → queue."""

    def __init__(self, chunk_paths: list[str], prefetch_count: int = 3):
        self._paths = chunk_paths
        self._prefetch_count = prefetch_count
        self._results_dict = {}
        self._next_yield_idx = 0
        self._proc_executor = concurrent.futures.ProcessPoolExecutor(max_workers=prefetch_count)
        self._thread_executor = concurrent.futures.ThreadPoolExecutor(max_workers=prefetch_count)
        self._ready_queue = queue.Queue(maxsize=prefetch_count)
        self._next_submit_idx = 0
        self._total_chunks = len(chunk_paths)

        for _ in range(min(prefetch_count, self._total_chunks)):
            self._submit_next_process()

    def _submit_next_process(self):
        if self._next_submit_idx < self._total_chunks:
            path = self._paths[self._next_submit_idx]
            idx = self._next_submit_idx
            self._next_submit_idx += 1
            future_proc = self._proc_executor.submit(_load_chunk_worker, path, idx)
            future_proc.add_done_callback(self._on_process_done)

    def _on_process_done(self, future):
        try:
            target_path, loaded_idx, proc_time, is_temp = future.result()
            self._thread_executor.submit(
                self._thread_loader_worker, target_path, loaded_idx, proc_time, is_temp
            )
        except Exception as e:
            print(f"[Prefetcher Error in Stage 1] {e}", file=sys.stderr)

    def _thread_loader_worker(self, target_path: str, idx: int, proc_time: float, is_temp: bool):
        t_start = time.time()
        try:
            chunk_data = torch.load(target_path, map_location="cpu", weights_only=True)
            if is_temp and os.path.exists(target_path):
                os.remove(target_path)
            t_end = time.time()
            self._ready_queue.put((idx, chunk_data, proc_time, t_end - t_start))
        except Exception as e:
            print(f"[Prefetcher Error in Stage 2] {e}", file=sys.stderr)

    def __iter__(self):
        for idx in range(self._total_chunks):
            t0 = time.time()
            print(f"\n[Prefetcher] Waiting for chunk {idx+1}/{self._total_chunks} to be ready...")
            while idx not in self._results_dict:
                ready_idx, chunk_data, proc_time, load_time = self._ready_queue.get()
                self._results_dict[ready_idx] = (chunk_data, proc_time, load_time)
            chunk_data, proc_time, load_time = self._results_dict.pop(idx)
            t1 = time.time()
            print(f"[Prefetcher] <- zstd: {proc_time:.2f}s | load: {load_time:.2f}s")
            print(f"[Prefetcher] Chunk {idx+1} ready (GPU blocked for: {t1 - t0:.3f}s).")
            self._submit_next_process()
            yield chunk_data

        self._proc_executor.shutdown(wait=False)
        self._thread_executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LayerAttentionHead(nn.Module):
    def __init__(self, hidden_size: int, num_stored_layers: int, pooling: str = "last"):
        super().__init__()
        self.pooling = pooling
        self.hidden_size = hidden_size
        self.layer_weights = nn.Parameter(torch.ones(num_stored_layers, dtype=torch.float32))

        if pooling == "attention":
            self.token_attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.Tanh(),
                nn.Linear(hidden_size // 4, 1),
            )
        elif pooling == "cnn_attention":
            # 1D-CNN over the subword sequence to capture morphological pairs
            self.cnn = nn.Sequential(
                nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=2, padding=1),
                nn.GELU()
            )
            # Standard attention over the CNN outputs
            self.token_attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.Tanh(),
                nn.Linear(hidden_size // 4, 1),
            )
        elif pooling == "global_query":
            self.global_query = nn.Parameter(
                torch.randn(hidden_size) / math.sqrt(hidden_size)
            )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, stacked: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stacked:        [B, L, S, H]  float16 on device
            attention_mask: [B, S]        bool on device
        Returns:
            [B, H]  float32
        """
        weights = torch.softmax(self.layer_weights, dim=0)
        fused = (stacked.float() * weights.view(1, -1, 1, 1)).sum(dim=1)

        if self.pooling == "last":
            seq_lengths = attention_mask.sum(dim=1).long() - 1
            pooled = fused[torch.arange(fused.size(0), device=fused.device), seq_lengths]

        elif self.pooling == "mean":
            mask_f = attention_mask.unsqueeze(-1).float()
            pooled = (fused * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-9)

        elif self.pooling == "attention":
            scores = self.token_attention(fused).squeeze(-1)
            scores = scores.masked_fill(~attention_mask, float("-inf"))
            attn_w = torch.softmax(scores, dim=1).unsqueeze(-1)
            pooled = (fused * attn_w).sum(dim=1)

        elif self.pooling == "cnn_attention":
            # fused is [B, S, H] -> Conv1d expects [B, C, L] where C=Channels(H), L=Length(S)
            fused_transposed = fused.transpose(1, 2)
            # padding=1 and kernel=2 will output Length S+1, we slice to get S
            cnn_out = self.cnn(fused_transposed)[:, :, :-1] # [B, H, S]
            cnn_fused = cnn_out.transpose(1, 2) # [B, S, H]
            
            # Apply standard attention pooling over the CNN outputs
            scores = self.token_attention(cnn_fused).squeeze(-1)
            scores = scores.masked_fill(~attention_mask, float("-inf"))
            attn_w = torch.softmax(scores, dim=1).unsqueeze(-1)
            pooled = (cnn_fused * attn_w).sum(dim=1)

        elif self.pooling == "global_query":
            scale = math.sqrt(self.hidden_size)
            scores = torch.einsum("bsh,h->bs", fused, self.global_query.float()) / scale
            scores = scores.masked_fill(~attention_mask, float("-inf"))
            attn_w = torch.softmax(scores, dim=1).unsqueeze(-1)
            pooled = (fused * attn_w).sum(dim=1)

        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return self.mlp(pooled.float()).to(torch.bfloat16)  # [B, H] float32  ← FIX: removed .to(bfloat16)


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

_VAL_LOADERS_CACHE = []


@torch.no_grad()
def evaluate(head, val_chunk_paths, target_embeddings, device,
             batch_size, loss_type, loss_kwargs, keep_in_ram, fast_metrics):
    """
    Returns dict of averaged metrics over all val batches.
    """
    global _VAL_LOADERS_CACHE
    head.eval()
    accum = {}
    total_batches = 0

    if keep_in_ram:
        if not _VAL_LOADERS_CACHE:
            print("\n[Eval] Loading and caching validation chunks in RAM...")
            for path in val_chunk_paths:
                chunk = _load_chunk(path)
                ds = CachedChunkDataset(chunk, target_embeddings, use_all_splits=False)
                loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                    num_workers=0, pin_memory=True)
                _VAL_LOADERS_CACHE.append(loader)
        loaders = _VAL_LOADERS_CACHE
    else:
        loaders = []
        for path in val_chunk_paths:
            chunk = _load_chunk(path)
            ds = CachedChunkDataset(chunk, target_embeddings, use_all_splits=False)
            loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=False,
                                      num_workers=0, pin_memory=True))

    for loader in loaders:
        for batch in loader:
            stacked = batch["hs"].to(device)
            mask = batch["mask"].to(device)
            tids = batch["tid"].to(device).long()

            pred = head(stacked, mask)
            metrics_dict = fast_metrics.compute_batch_metrics(pred, tids)
            loss = fast_metrics.compute_loss(pred, tids, loss_type=loss_type, **loss_kwargs)
            metrics_dict["loss"] = loss.detach()

            for k, v in metrics_dict.items():
                accum[k] = accum.get(k, 0.0) + v.item()
            total_batches += 1

    head.train()
    return {k: v / max(total_batches, 1) for k, v in accum.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pooling", type=str, default="last",
                        choices=["last", "mean", "attention", "global_query"])
    parser.add_argument("--all-splits", action="store_true",
                        help="WeightedRandomSampler (50/30/20 len distribution)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Stop after this many optimizer steps (0 = disabled)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha-mse", type=float, default=1000.0,
                        help="Weight of the MSE term in legacy composite/contrastive losses")
    parser.add_argument("--reldist-ratio", type=float, default=0.5)
    parser.add_argument("--loss-type", type=str, choices=[
        # Tier 1 — baselines
        "mse", "cosine", "centered_cosine", "cosine_norm", "pca_weighted_mse",
        "mse_norm", "mse_plus_cosine",
        # Tier 2 — functional
        "soft_logit_kl", "multi_tau_kl", "contrastive_logit",
        # Tier 3 — composite
        "geometric_aware", "full_spectrum",
        # Legacy
        "mse_cosine", "reldist", "reldist_mse_cosine", "reldist_mse_cosine_mult",
        "triplet_composite",
        "infonce_cosine", "infonce_mse",
        "infonce_cosine_anchored", "infonce_margin_anchored",
        "infonce_composite_anchored", "infonce_composite_margin_anchored",
    ], default="mse", help="Loss function to optimize")
    parser.add_argument("--loss-kwargs", type=str, default="{}",
                        help='JSON dict of extra kwargs for loss, e.g. \'{"tau": 0.5}\'')
    parser.add_argument("--cache-dir", type=str, 
                        default="cache")
    parser.add_argument("--model-path", type=str, 
                        default="/workdir/models/Qwen3.5-2B-Base", help="Path to base model (requires external storage due to size)")
    
    parser.add_argument("--results-dir", type=str, 
                        default="results")
    parser.add_argument("--eval-every", type=int, default=500,
                        help="Evaluate on full val set every N optimizer steps")
    parser.add_argument("--keep-val-in-ram", action="store_true",
                        help="Load validation chunks into RAM once and keep them")
    parser.add_argument("--prefetch-count", type=int, default=4,
                        help="Number of background processes for zstd decompression")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to a pre-trained head checkpoint (.pt) to initialize weights from.")
    args = parser.parse_args()

    # Parse loss kwargs
    loss_kwargs = json.loads(args.loss_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Device: {device} | pooling: {args.pooling} | lr: {args.lr} | bs: {args.batch_size}")
    print(f"Loss: {args.loss_type} | kwargs: {loss_kwargs}")

    # ------------------------------------------------------------------
    # Load target embeddings
    # ------------------------------------------------------------------
    print(f"Loading target embeddings from {args.model_path} (CPU)...")
    llm = load_causal_lm(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    target_embeddings = llm.get_input_embeddings().weight.detach()
    del llm
    torch.cuda.empty_cache()

    # Always move to device
    target_embeddings = target_embeddings.to(device)
    print(f"Target embeddings: {target_embeddings.shape} on {target_embeddings.device}")

    # ------------------------------------------------------------------
    # Build run suffix for file naming
    # ------------------------------------------------------------------
    suffix = f"h1_1_{args.pooling}"
    if args.all_splits:
        suffix += "_all_splits"
    suffix += f"_{args.loss_type}"
    suffix += f"_lr{args.lr}"
    suffix += f"_e{args.epochs}"
    if args.loss_kwargs != "{}":
        # Compact representation for filename
        suffix += f"_kw{args.loss_kwargs.replace(' ', '').replace('\"', '')}"
    print(f"Run suffix: {suffix}")

    # ------------------------------------------------------------------
    # Load cache meta
    # ------------------------------------------------------------------
    train_meta_path = os.path.join(args.cache_dir, "train", "meta.json")
    val_meta_path = os.path.join(args.cache_dir, "val", "meta.json")

    with open(train_meta_path) as f:
        train_meta = json.load(f)
    with open(val_meta_path) as f:
        val_meta = json.load(f)

    num_stored_layers = train_meta["num_stored_layers"]
    hidden_size = train_meta["hidden_size"]

    train_chunks = sorted([
        os.path.join(args.cache_dir, "train", fn)
        for fn in os.listdir(os.path.join(args.cache_dir, "train"))
        if fn.startswith("chunk_") and (fn.endswith(".pt") or fn.endswith(".pt.zst"))
    ])
    val_chunks = sorted([
        os.path.join(args.cache_dir, "val", fn)
        for fn in os.listdir(os.path.join(args.cache_dir, "val"))
        if fn.startswith("chunk_") and (fn.endswith(".pt") or fn.endswith(".pt.zst"))
    ])[:1]
    print(f"Train chunks: {len(train_chunks)} | Val chunks: {len(val_chunks)}")
    print(f"Layers stored: {num_stored_layers} | Hidden size: {hidden_size}")

    # ------------------------------------------------------------------
    # Build head & optimizer
    # ------------------------------------------------------------------
    head = LayerAttentionHead(hidden_size, num_stored_layers, pooling=args.pooling).to(device)
    head.mlp.to(torch.float32)

    if args.resume_from:
        print(f"Loading pre-trained head weights from: {args.resume_from}")
        state_dict = torch.load(args.resume_from, map_location=device, weights_only=True)
        head.load_state_dict(state_dict, strict=True)
        print("Successfully loaded pre-trained weights.")

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)

    examples_per_epoch = train_meta["total"]
    steps_per_epoch = math.ceil(examples_per_epoch / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps > 0:
        total_steps = args.max_steps
    warmup_steps = max(1, int(0.05 * total_steps))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print(f"Total steps: {total_steps} | Warmup: {warmup_steps} | Epochs: {args.epochs}")

    # ------------------------------------------------------------------
    # Initialize FastMetrics
    # ------------------------------------------------------------------
    fast_metrics = FastMetrics(
        target_embeddings,
        alpha_mse=args.alpha_mse,
        reldist_ratio=args.reldist_ratio,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    log = {
        "args": vars(args),
        "train_losses_step": [],
        "val_steps": [],
        "lrs": [],
        "metrics": {},
    }
    global_step = 0
    LOG_EVERY = 10

    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")

        prefetcher = ChunkPrefetcher(train_chunks, prefetch_count=args.prefetch_count)
        epoch_pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}", dynamic_ncols=True)

        for chunk_idx, chunk in enumerate(prefetcher):
            loader = make_loader(chunk, target_embeddings, args.batch_size, args.all_splits)

            head.train()
            for batch in loader:
                stacked = batch["hs"].to(device)
                mask = batch["mask"].to(device)
                tids = batch["tid"].to(device).long()

                optimizer.zero_grad()

                pred = head(stacked, mask)

                loss = fast_metrics.compute_loss(
                    pred, tids, loss_type=args.loss_type, **loss_kwargs
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                global_step += 1
                lr_now = scheduler.get_last_lr()[0]

                if global_step % LOG_EVERY == 0:
                    log["train_losses_step"].append(loss.item())
                    log["lrs"].append(lr_now)

                epoch_pbar.set_postfix({
                    "loss": f"{loss.item():.6f}",
                    "lr": f"{lr_now:.2e}",
                    "chunk": f"{chunk_idx+1}/{len(train_chunks)}"
                })
                epoch_pbar.update(1)

                # Periodic validation
                if global_step % args.eval_every == 0:
                    val_metrics = evaluate(
                        head, val_chunks, target_embeddings, device,
                        args.batch_size, args.loss_type, loss_kwargs,
                        args.keep_val_in_ram, fast_metrics,
                    )

                    # Log EVERYTHING to JSON
                    for k, v in val_metrics.items():
                        log["metrics"].setdefault(k, []).append(v)
                    log["val_steps"].append(global_step)
                
                    # Print only key metrics to console
                    key_metrics = [
                        "loss", "logit_mrr", "logit_top1_acc", "logit_top5_acc",
                        "centered_cos_dist", "norm_error", "soft_kl_0.5", "mse"
                    ]
                    print(f"\n  [step {global_step}] " + " | ".join(
                        f"{k}={val_metrics[k]:.6f}" for k in key_metrics if k in val_metrics
                    ))

                    # Save logs
                    with open(os.path.join(args.results_dir, f"{suffix}_logs.json"), "w") as f:
                        json.dump(log, f)

                if args.max_steps > 0 and global_step >= args.max_steps:
                    print(f"\nReached max_steps={args.max_steps}. Stopping.")
                    break

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        epoch_pbar.close()

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    final_path = os.path.join(args.results_dir, f"{suffix}_head_final.pt")
    torch.save(head.state_dict(), final_path)
    print(f"\nSaved final head → {final_path}")

    # Final validation
    val_metrics = evaluate(
        head, val_chunks, target_embeddings, device,
        args.batch_size, args.loss_type, loss_kwargs,
        args.keep_val_in_ram, fast_metrics,
    )
    print("Final metrics: " + " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))

    for k, v in val_metrics.items():
        log["metrics"].setdefault(k, []).append(v)
    log["val_steps"].append(global_step)

    with open(os.path.join(args.results_dir, f"{suffix}_logs.json"), "w") as f:
        json.dump(log, f)

    print("Done.")


if __name__ == "__main__":
    main()