"""Freeze/unfreeze utilities + embed gradient hooks.

Provides composable primitives for freezing model parameters:
- freeze_all_except: freeze everything except named modules
- freeze_by_pattern: freeze by regex pattern on parameter names
- register_embed_freeze_hook: gradient zeroing hook for embed rows (CPT pattern)
- get_trainable_summary: count trainable vs total parameters
"""

import re
from typing import List, Optional, Set

import torch
import torch.nn as nn


def freeze_all_except(model: nn.Module, modules_to_keep: List[str]) -> None:
    """Freeze all parameters except those belonging to named modules.

    Args:
        model: The model to modify.
        modules_to_keep: Substring matches for module names to keep trainable.
            E.g. ["embed_tokens", "lm_head"] keeps both embedding layers trainable.
    """
    for name, param in model.named_parameters():
        if any(k in name for k in modules_to_keep):
            param.requires_grad = True
        else:
            param.requires_grad = False


def freeze_by_pattern(model: nn.Module, pattern: str) -> None:
    """Freeze parameters whose names match a regex pattern.

    Args:
        model: The model to modify.
        pattern: Regex pattern. Parameters with matching names are frozen.
    """
    regex = re.compile(pattern)
    for name, param in model.named_parameters():
        if regex.search(name):
            param.requires_grad = False


def register_embed_freeze_hook(model: nn.Module, freeze_idx: int) -> None:
    """Register gradient hooks that zero out gradients for embed rows < freeze_idx.

    From CPT script: prevents new tokens from being modified by gradients of
    old tokens during continued pretraining. Applied to both input and output
    embeddings (if not tied).

    Args:
        model: The model.
        freeze_idx: Token IDs below this value have their gradients zeroed.
    """

    def mask_grad_hook(grad: torch.Tensor, limit: int = freeze_idx) -> torch.Tensor:
        actual_limit = min(limit, grad.shape[0])
        with torch.no_grad():
            grad[:actual_limit, :] = 0.0
        return grad

    embeds = model.get_input_embeddings()
    embeds.weight.register_hook(mask_grad_hook)

    lm_head = model.get_output_embeddings()
    if lm_head is not None and lm_head.weight is not embeds.weight:
        lm_head.weight.register_hook(mask_grad_hook)


def get_trainable_summary(model: nn.Module) -> dict:
    """Count trainable vs total parameters.

    Returns:
        dict with keys: trainable, total, percentage
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "percentage": 100.0 * trainable / total if total > 0 else 0.0,
    }
