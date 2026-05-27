"""
18_precompute_hidden_states.py — E1: Precompute LLM hidden states to disk.

Saves stacked hidden states [num_stored_layers, seq_len, hidden_size] in bfloat16
for every example in train_bpe.json and val_bpe.json as chunked .pt files.

Storage layout:
  cache_dir/
    train/
      chunk_0000.pt[.zst]  → dict with keys:
                               'hidden_states': Tensor [N, num_stored_layers, max_seqlen, hidden_size] bfloat16
                               'attention_mask': Tensor [N, max_seqlen] bool
                               'target_ids': Tensor [N] int32
                               'seq_lens': Tensor [N] uint8  (actual seqlen per example)
                               'target_strs': list[str]
                               'target_decoded': list[str]
                               'fragmented_strs': list[list[str]]
                               'fragmented_decoded': list[list[str]]
      chunk_0001.pt[.zst]
      ...
      meta.json
    val/
      ...

Usage:
  python 18_precompute_hidden_states.py [--chunk-size 10000] [--layers-from-top 24]
                                         [--cache-dir cache]
                                         [--split train|val|both] [--batch-size 512]
                                         [--compress] [--compress-level 3]
"""

import os
import json
import math
import argparse
import io
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoConfig
from tqdm import tqdm

from ruadapt.utils.model_utils import load_causal_lm

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_chunk(path: str, compressed: bool) -> dict:
    """Load a chunk file, decompressing if needed."""
    if compressed:
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        with open(path, "rb") as f:
            raw = dctx.decompress(f.read())
        return torch.load(io.BytesIO(raw), weights_only=True)
    else:
        return torch.load(path, weights_only=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SubwordDatasetBPE(Dataset):
    def __init__(self, data_file: str):
        with open(data_file) as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} examples from {data_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "target_id": item["target_id"],
            "target_str": item.get("target_str", ""),
            "target_decoded": item.get("target_decoded", ""),
            "fragmented_ids": torch.tensor(item["fragmented_ids"], dtype=torch.long),
            "fragmented_strs": item.get("fragmented_strs", []),
            "fragmented_decoded": item.get("fragmented_decoded", []),
        }


def collate_fn(batch):
    max_len = max(len(item["fragmented_ids"]) for item in batch)
    padded, masks, target_ids, seq_lens = [], [], [], []
    t_strs, t_decs, f_strs, f_decs = [], [], [], []
    for item in batch:
        ids = item["fragmented_ids"]
        sl = len(ids)
        pad = max_len - sl
        padded.append(torch.cat([ids, torch.zeros(pad, dtype=torch.long)]))
        masks.append(torch.cat([torch.ones(sl, dtype=torch.bool),
                                 torch.zeros(pad, dtype=torch.bool)]))
        target_ids.append(item["target_id"])
        seq_lens.append(sl)
        t_strs.append(item["target_str"])
        t_decs.append(item["target_decoded"])
        f_strs.append(item["fragmented_strs"])
        f_decs.append(item["fragmented_decoded"])
    return {
        "input_ids":      torch.stack(padded),
        "attention_mask": torch.stack(masks),
        "target_ids":     torch.tensor(target_ids, dtype=torch.int32),
        "seq_lens":       torch.tensor(seq_lens, dtype=torch.long),
        "target_strs":    t_strs,
        "target_decoded": t_decs,
        "fragmented_strs": f_strs,
        "fragmented_decoded": f_decs,
    }


# ---------------------------------------------------------------------------
# Main precompute logic
# ---------------------------------------------------------------------------

def precompute(split: str, data_file: str, cache_dir: str, model, layer_indices: list,
               chunk_size: int, batch_size: int, device: torch.device,
               compress: bool = False, compress_level: int = 3):
    """Iterate over dataset, run LLM forward, accumulate hidden states, flush chunks."""
    if compress and not ZSTD_AVAILABLE:
        raise RuntimeError("zstandard not installed. Run: pip install zstandard")

    ext = ".pt.zst" if compress else ".pt"
    out_dir = os.path.join(cache_dir, split)
    os.makedirs(out_dir, exist_ok=True)

    compressor = zstd.ZstdCompressor(level=compress_level, threads=-1) if compress else None

    dataset = SubwordDatasetBPE(data_file)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )

    num_stored_layers = len(layer_indices)
    total = len(dataset)
    num_chunks = math.ceil(total / chunk_size)

    # Determine global MAX_SEQLEN from dataset
    MAX_SEQLEN = max(len(item["fragmented_ids"]) for item in dataset.data)
    print(f"MAX_SEQLEN from data: {MAX_SEQLEN}")

    # Determine hidden_size from model
    with torch.no_grad():
        probe = model.model(
            input_ids=torch.tensor([[1]], device=device),
            output_hidden_states=True,
        )
    HIDDEN_SIZE = probe.hidden_states[0].shape[-1]
    print(f"HIDDEN_SIZE: {HIDDEN_SIZE}")
    del probe

    # Resume state: check existing chunk files
    meta_path = os.path.join(out_dir, "meta.json")
    done_chunks = set()
    for ci in range(num_chunks):
        p = os.path.join(out_dir, f"chunk_{ci:04d}{ext}")
        if os.path.exists(p):
            done_chunks.add(ci)

    if done_chunks and not os.path.exists(meta_path):
        last_chunk = max(done_chunks)
        done_chunks.remove(last_chunk)
        # Remove the potentially corrupted file
        corrupt_path = os.path.join(out_dir, f"chunk_{last_chunk:04d}{ext}")
        if os.path.exists(corrupt_path):
            os.remove(corrupt_path)
        print(f"Removed potentially incomplete chunk_{last_chunk:04d}{ext} from resume state.")

    if done_chunks:
        print(f"Resuming: {len(done_chunks)}/{num_chunks} chunks already done.")

    # ===== PRE-ALLOCATED CHUNK BUFFER =====
    # Shape: [chunk_size, num_stored_layers, MAX_SEQLEN, HIDDEN_SIZE], bfloat16
    buf_nbytes = chunk_size * num_stored_layers * MAX_SEQLEN * HIDDEN_SIZE * 2
    print(f"Pre-allocating chunk buffer: [{chunk_size}, {num_stored_layers}, {MAX_SEQLEN}, {HIDDEN_SIZE}] bf16")
    print(f"  Buffer size: {buf_nbytes / 1e9:.2f} GB")

    chunk_hs = torch.zeros(chunk_size, num_stored_layers, MAX_SEQLEN, HIDDEN_SIZE, dtype=torch.bfloat16)

    # Metadata buffers (lists for non-tensor data)
    buf_tids = []
    buf_slens = []
    buf_tstrs = []
    buf_tdecs = []
    buf_fstrs = []
    buf_fdecs = []

    chunk_idx = 0    # index of the chunk we are currently filling
    buf_pos = 0      # number of examples written into current chunk buffer

    def flush_chunk(ci: int, actual_size: int):
        """Save the current chunk buffer to disk."""
        nonlocal buf_pos

        if actual_size == 0:
            return

        if ci in done_chunks:
            print(f"  Chunk {ci:04d} already exists, skipping save.")
            # Clear buffers
            buf_tids.clear(); buf_slens.clear()
            buf_tstrs.clear(); buf_tdecs.clear()
            buf_fstrs.clear(); buf_fdecs.clear()
            # Zero out the buffer for reuse
            chunk_hs[:actual_size] = 0
            buf_pos = 0
            return

        t0 = time.time()

        # Clone the filled portion of the buffer
        hs_tensor = chunk_hs[:actual_size].clone()

        t1 = time.time()

        # Generate attention_mask from seq_lens (vectorized)
        seq_lens_t = torch.tensor(buf_slens, dtype=torch.long)
        mask_tensor = torch.arange(MAX_SEQLEN).unsqueeze(0) < seq_lens_t.unsqueeze(1)  # [N, MAX_SEQLEN] bool

        t2 = time.time()

        chunk_data = {
            "hidden_states":     hs_tensor,
            "attention_mask":    mask_tensor,
            "target_ids":        torch.tensor(buf_tids, dtype=torch.int32),
            "seq_lens":          torch.tensor(buf_slens, dtype=torch.uint8),
            "target_strs":       list(buf_tstrs),
            "target_decoded":    list(buf_tdecs),
            "fragmented_strs":   list(buf_fstrs),
            "fragmented_decoded": list(buf_fdecs),
        }

        t3 = time.time()

        path = os.path.join(out_dir, f"chunk_{ci:04d}{ext}")
        if compress:
            buf_io = io.BytesIO()
            torch.save(chunk_data, buf_io)
            raw_bytes = buf_io.getvalue()
            compressed_bytes = compressor.compress(raw_bytes)
            with open(path, "wb") as f:
                f.write(compressed_bytes)
            size_mb = len(compressed_bytes) / 1e6
            ratio = len(raw_bytes) / len(compressed_bytes)
            print(f"  Saved chunk {ci:04d} ({actual_size} examples, {size_mb:.1f} MB zstd, "
                  f"ratio={ratio:.2f}x) → {path}")
        else:
            torch.save(chunk_data, path)
            size_mb = os.path.getsize(path) / 1e6
            print(f"  Saved chunk {ci:04d} ({actual_size} examples, {size_mb:.1f} MB) → {path}")

        t4 = time.time()
        print(f"  Timings: clone={t1-t0:.2f}s mask={t2-t1:.3f}s "
              f"dict={t3-t2:.3f}s save={t4-t3:.2f}s total={t4-t0:.2f}s")

        # Clear metadata buffers
        buf_tids.clear(); buf_slens.clear()
        buf_tstrs.clear(); buf_tdecs.clear()
        buf_fstrs.clear(); buf_fdecs.clear()

        # Zero out used portion of chunk_hs for next chunk
        chunk_hs[:actual_size] = 0
        buf_pos = 0

    # ===== MAIN LOOP =====
    model.eval()
    pbar = tqdm(loader, desc=f"Precompute [{split}]", unit="batch")

    with torch.no_grad():
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_ids = batch["target_ids"]       # CPU, int32
            seq_lens = batch["seq_lens"]           # CPU, long
            target_strs = batch["target_strs"]
            target_decoded = batch["target_decoded"]
            frag_strs = batch["fragmented_strs"]
            frag_decoded = batch["fragmented_decoded"]

            # Forward pass
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

            # Extract selected layers → [B, num_stored_layers, S_batch, H]
            # S_batch = max seqlen in this batch (from collate_fn padding)
            selected = torch.stack(
                [outputs.hidden_states[i].to(torch.bfloat16).cpu() for i in layer_indices],
                dim=0,  # [L, B, S_batch, H]
            ).permute(1, 0, 2, 3).contiguous()  # [B, L, S_batch, H]

            B = selected.shape[0]
            S_batch = selected.shape[2]

            # Zero out padding positions (vectorized, no Python loop)
            # seq_lens[b] = actual number of tokens for example b
            # Positions >= seq_lens[b] contain garbage from model (not zeros)
            # We must zero them to match original behavior
            valid_mask = torch.arange(S_batch).unsqueeze(0) < seq_lens.unsqueeze(1)  # [B, S_batch] bool
            # Expand to [B, 1, S_batch, 1] for broadcasting with [B, L, S_batch, H]
            valid_mask_expanded = valid_mask[:, None, :, None]  # [B, 1, S_batch, 1]
            selected.mul_(valid_mask_expanded)  # in-place zero out padding positions

            # Now write selected into the pre-allocated chunk buffer.
            # Handle the case where this batch crosses a chunk boundary.
            written = 0  # how many examples from this batch we have processed

            while written < B:
                space_left = chunk_size - buf_pos
                to_write = min(B - written, space_left)

                # Slice of this batch to write
                sel_slice = selected[written : written + to_write]  # [to_write, L, S_batch, H]

                # Write into chunk buffer (single contiguous slice assignment)
                chunk_hs[buf_pos : buf_pos + to_write, :, :S_batch, :] = sel_slice

                # Append metadata
                for b in range(written, written + to_write):
                    buf_tids.append(int(target_ids[b].item()))
                    buf_slens.append(int(seq_lens[b].item()))
                    buf_tstrs.append(target_strs[b])
                    buf_tdecs.append(target_decoded[b])
                    buf_fstrs.append(frag_strs[b])
                    buf_fdecs.append(frag_decoded[b])

                buf_pos += to_write
                written += to_write

                # If chunk is full, flush it
                if buf_pos >= chunk_size:
                    flush_chunk(chunk_idx, chunk_size)
                    chunk_idx += 1

            pbar.set_postfix({"chunk": f"{chunk_idx}/{num_chunks}", "buf": buf_pos})

    # Flush the last partial chunk
    if buf_pos > 0:
        flush_chunk(chunk_idx, buf_pos)

    # Write meta.json
    meta = {
        "split":             split,
        "total":             total,
        "chunk_size":        chunk_size,
        "num_chunks":        num_chunks,
        "num_stored_layers": num_stored_layers,
        "layer_indices":     layer_indices,
        "hidden_size":       HIDDEN_SIZE,
        "max_seqlen":        MAX_SEQLEN,
        "compressed":        compress,
        "compress_level":    compress_level if compress else None,
        "chunk_ext":         ext,
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{split}] Done. {num_chunks} chunks, meta → {meta_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Precompute hidden states for tokenizer init research")
    parser.add_argument("--model-path",    type=str, default="/workdir/models/Qwen3.5-2B-Base", help="Path to base model (requires external storage due to size)")
    parser.add_argument("--train-file",    type=str, default="data/train_bpe_shuffled.json")
    parser.add_argument("--val-file",      type=str, default="data/val_bpe_shuffled.json")
    parser.add_argument("--cache-dir",     type=str, default="cache")
    parser.add_argument("--split",         type=str, choices=["train", "val", "both"], default="both")
    parser.add_argument("--chunk-size",    type=int, default=10000,
                        help="Number of examples per .pt chunk file")
    parser.add_argument("--batch-size",    type=int, default=512,
                        help="LLM inference batch size")
    parser.add_argument("--layers-from-top", type=int, default=24,
                        help="Store this many transformer layers counting from the output (top). "
                             "embed_tokens (index 0) is ALWAYS stored in addition.")
    parser.add_argument("--compress",      action="store_true",
                        help="Compress chunks with zstd (requires: pip install zstandard).")
    parser.add_argument("--compress-level", type=int, default=3,
                        help="zstd compression level (1=fast, 3=default, 9=max). Default: 3")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {args.model_path}...")
    try:
        llm = load_causal_lm(
            args.model_path,
            torch_dtype=torch.bfloat16,
            output_hidden_states=True,
            attn_implementation="flash_attention_2",
        ).to(device)
    except Exception:
        print("flash_attention_2 unavailable, falling back to sdpa")
        llm = load_causal_lm(
            args.model_path,
            torch_dtype=torch.bfloat16,
            output_hidden_states=True,
            attn_implementation="sdpa",
        ).to(device)

    llm.eval()
    for p in llm.parameters():
        p.requires_grad_(False)

    # Probe model structure
    with torch.no_grad():
        probe = llm.model(
            input_ids=torch.tensor([[1, 2, 3]], device=device),
            output_hidden_states=True,
        )
    num_total_hs = len(probe.hidden_states)
    num_transformer_layers = num_total_hs - 1
    print(f"Model: {num_transformer_layers} transformer layers + 1 embed_tokens = {num_total_hs} hidden states")
    del probe

    # Determine layer indices to store
    layers_from_top = min(args.layers_from_top, num_transformer_layers)
    transformer_indices = list(range(1, num_total_hs))
    top_transformer_indices = transformer_indices[-layers_from_top:]
    layer_indices = [0] + top_transformer_indices  # always include embed_tokens
    print(f"Storing {len(layer_indices)} layers: embed_tokens(0) + transformer indices "
          f"{top_transformer_indices[0]}..{top_transformer_indices[-1]}")

    # Determine splits to process
    splits_to_run = ["train", "val"] if args.split == "both" else [args.split]

    # Print storage estimates
    total_examples = 0
    total_tokens = 0
    for s in splits_to_run:
        data_file = args.train_file if s == "train" else args.val_file
        if os.path.exists(data_file):
            with open(data_file, "r") as f:
                data = json.load(f)
                total_examples += len(data)
                total_tokens += sum(len(item["fragmented_ids"]) for item in data)

    if total_examples > 0:
        avg_seqlen = total_tokens / total_examples
        hidden_size = probe.hidden_states[0].shape[-1] if 'probe' in dir() else 1536
        # Use MAX_SEQLEN for estimate since we pad to it
        max_sl_est = max(len(item["fragmented_ids"]) for item in data)
        est_gb_raw = total_examples * len(layer_indices) * max_sl_est * hidden_size * 2 / 1e9
        est_gb_zstd = est_gb_raw / 2.0
        print(f"Dataset stats: {total_examples} examples, avg_seqlen={avg_seqlen:.2f}")
        print(f"Estimated cache size: {est_gb_raw:.1f} GB raw"
              + (f" / {est_gb_zstd:.1f} GB with zstd-{args.compress_level}" if args.compress else ""))

    # Process each split
    for s in splits_to_run:
        data_file = args.train_file if s == "train" else args.val_file
        print(f"\n{'='*60}")
        print(f"Processing split: {s} ({data_file})")
        print(f"{'='*60}")
        precompute(
            split=s,
            data_file=data_file,
            cache_dir=args.cache_dir,
            model=llm,
            layer_indices=layer_indices,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
            device=device,
            compress=args.compress,
            compress_level=args.compress_level,
        )

    print("\nPrecompute complete.")


if __name__ == "__main__":
    main()