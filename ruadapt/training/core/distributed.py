"""Distributed training setup: process group init, model wrapping.

Supports FSDP (default), DDP, and DeepSpeed (via HF Trainer).
"""

import os
from typing import Optional

import torch
import torch.distributed as dist


def init_distributed() -> None:
    """Initialize distributed process group from environment variables.

    Expected env vars (set by torchrun): LOCAL_RANK, RANK, WORLD_SIZE.
    If already initialized, this is a no-op.
    If WORLD_SIZE is not set, assumes single-GPU and skips init.
    """
    if dist.is_initialized():
        return

    # Single-GPU mode: no torchrun env vars → skip distributed init
    if "WORLD_SIZE" not in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="nccl")


def is_main_process() -> bool:
    """Check if current process is rank 0."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def main_process_first(fn):
    """Run `fn` on rank 0 first, then on the other ranks after a barrier.

    For dataset preparation: rank 0 computes and populates the HF-datasets
    cache, the other ranks wait and read from cache instead of recomputing
    the same deterministic maps in parallel. Single-process mode runs fn directly.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return fn()
    if dist.get_rank() == 0:
        result = fn()
        dist.barrier()
        return result
    dist.barrier()
    return fn()


def wrap_model(model, config) -> object:
    """Wrap model for distributed training based on config.

    Strategy selection:
    - If config.training.deepspeed is set → no wrapping here (HF Trainer handles it)
    - If config.training.fsdp is set → FSDP wrapping
    - Otherwise → DDP wrapping (or no wrapping for single GPU)

    Args:
        model: The model to wrap.
        config: MainConfig (or any object with .training attribute).

    Returns:
        Wrapped model (or original if single GPU / DeepSpeed).
    """
    if not dist.is_initialized():
        return model

    training_args = config.training

    # DeepSpeed: HF Trainer handles wrapping internally
    if getattr(training_args, "deepspeed", None):
        return model

    # FSDP: already configured via TrainingArguments.fsdp
    if getattr(training_args, "fsdp", None):
        # FSDP wrapping is handled by HF Trainer via accelerate
        # No manual wrapping needed when using TrainingArguments.fsdp
        return model

    # DDP fallback
    device_ids = [int(os.environ.get("LOCAL_RANK", 0))]
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=device_ids,
        output_device=device_ids[0],
    )
    return model


def cleanup_distributed() -> None:
    """Destroy distributed process group if initialized."""
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
