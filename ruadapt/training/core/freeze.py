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
    Always freezes vision/visual parts of VLM models unless explicitly included in modules_to_keep.

    Args:
        model: The model to modify.
        modules_to_keep: Substring matches for module names to keep trainable.
            E.g. ["embed_tokens", "lm_head"] keeps both embedding layers trainable.
    """
    for name, param in model.named_parameters():
        # Safeguard: if parameter belongs to visual tower, freeze it unless explicitly unfrozen
        is_visual = any(v in name for v in ["visual", "vision_tower", "img_projection"])
        keep_visual = any(v in name for v in modules_to_keep if any(x in v for x in ["visual", "vision_tower", "img_projection"]))
        
        if is_visual and not keep_visual:
            param.requires_grad = False
            continue

        if any(k in name for k in modules_to_keep):
            param.requires_grad = True
        else:
            param.requires_grad = False


def freeze_by_pattern(model: nn.Module, pattern: str) -> None:
    """Freeze parameters whose names match a regex pattern.
    Always freezes vision/visual parts of VLM models as a safeguard.

    Args:
        model: The model to modify.
        pattern: Regex pattern. Parameters with matching names are frozen.
    """
    regex = re.compile(pattern)
    for name, param in model.named_parameters():
        is_visual = any(v in name for v in ["visual", "vision_tower", "img_projection"])
        if is_visual or regex.search(name):
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


def get_trainable_summary(model: nn.Module, freeze_idx: Optional[int] = None) -> dict:
    """Count trainable vs total parameters.
    
    If freeze_idx is provided, adjusts the trainable parameters count for embedding layers
    to exclude the frozen vocabulary rows, providing a correct representation for embeddings_partial.

    Returns:
        dict with keys: trainable, total, percentage
    """
    trainable = 0
    total = 0
    
    embed_input = model.get_input_embeddings()
    embed_output = model.get_output_embeddings()

    for name, param in model.named_parameters():
        p_total = param.numel()
        p_trainable = p_total if param.requires_grad else 0
        
        # Adjust count for embeddings_partial if freeze_idx is present
        if freeze_idx is not None and param.requires_grad:
            is_input_embed = embed_input is not None and param is embed_input.weight
            is_output_embed = embed_output is not None and param is embed_output.weight
            
            if is_input_embed or is_output_embed:
                # Embeddings are represented as (vocab_size, hidden_dim)
                vocab_size, hidden_dim = param.shape[0], param.shape[1]
                # Elements below freeze_idx are frozen by gradient hook
                frozen_rows = min(freeze_idx, vocab_size)
                p_trainable = (vocab_size - frozen_rows) * hidden_dim
                
        trainable += p_trainable
        total += p_total

    return {
        "trainable": trainable,
        "total": total,
        "percentage": 100.0 * trainable / total if total > 0 else 0.0,
    }
