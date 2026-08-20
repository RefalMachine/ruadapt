"""FP8 weight storage with bf16 compute (QLoRA-style) for frozen base models.

Loads an FP8-quantized checkpoint (HF finegrained fp8, block 128x128) and
replaces the forward of every quantized FP8Linear with an autograd-friendly
dequantize -> F.linear path. Weights stay in fp8 in memory (~half of bf16),
all compute happens in bf16. The frozen base needs no weight gradients, so
only the input-gradient path matters and it is exact bf16.

This makes gradient checkpointing optional: the memory it frees can outweigh
the recompute cost (see ANALYSYS.md, section 8).

Not supported: true FP8 compute (W8A8 forward) — HF's FP8Linear forward uses
Triton kernels without autograd (FineGrainedFP8HfQuantizer.is_trainable=False).
PEFT merge into fp8 base weights — load the base with
FineGrainedFP8Config(dequantize=True), merge adapters in bf16, then re-quantize
for serving with transformers' Fp8Quantize (see FP8.md, section 7).
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
PATCHED_ATTR = "_ruadapt_fp8_dequant"


def _is_fp8_quantized_linear(module: torch.nn.Module) -> bool:
    return (
        module.__class__.__name__ == "FP8Linear"
        and getattr(module, "weight", None) is not None
        and module.weight.dtype in FP8_DTYPES
        and getattr(module, "weight_scale_inv", None) is not None
    )


def dequantize_fp8_block(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block_size: Tuple[int, int] = (128, 128),
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize block-quantized fp8 weights to `dtype`.

    weight: (N, K) fp8; weight_scale_inv: (ceil(N/bn), ceil(K/bk)) float32.
    Per-tensor mode (block_size=None) expects a scalar scale.

    Compute happens in `dtype` (bf16 in training): fp8 -> bf16 cast is exact,
    fp32->bf16 scale rounding adds <=0.2% relative error on top of fp8's own
    quantization error. The fp32 path is kept for exactness in tests.
    """
    if block_size is None:
        return (weight.to(torch.float32) * weight_scale_inv).to(dtype)

    n, k = weight.shape
    bn, bk = block_size
    # copy=True: to() returns the same object when dtype matches, and the mul_
    # below would then destroy the source weight (e.g. a bf16 weight passed
    # through this path during LoRA merge on a dequantized base).
    w = weight.to(dtype, copy=True)
    scale = weight_scale_inv.to(torch.float32 if dtype == torch.float32 else dtype)
    if n % bn == 0 and k % bk == 0:
        w = w.view(n // bn, bn, k // bk, bk)
        scale = scale.view(n // bn, 1, k // bk, 1)
        w = w.mul_(scale)
        return w.view(n, k)
    scale = scale.repeat_interleave(bn, dim=0)[:n].repeat_interleave(bk, dim=1)[:, :k]
    return w.mul_(scale)


# Active dequant implementation; swapped to a torch.compile'd version (single
# fused memory pass instead of cast+mul as separate kernels) when enabled.
_ACTIVE_DEQUANT = dequantize_fp8_block
_COMPILED_DEQUANT_CACHE = None


def set_dequant_compiled(enabled: bool) -> None:
    """Compile the dequant op. CUDA-only; falls back to eager on failure."""
    global _ACTIVE_DEQUANT, _COMPILED_DEQUANT_CACHE
    if not enabled:
        _ACTIVE_DEQUANT = dequantize_fp8_block
        return
    if _COMPILED_DEQUANT_CACHE is None:
        try:
            _COMPILED_DEQUANT_CACHE = torch.compile(dequantize_fp8_block, dynamic=False)
        except Exception as e:  # pragma: no cover - defensive
            print(f"[fp8-storage] torch.compile(dequant) unavailable, using eager: {e}")
            _COMPILED_DEQUANT_CACHE = dequantize_fp8_block
    _ACTIVE_DEQUANT = _COMPILED_DEQUANT_CACHE


def _make_dequant_forward(module: torch.nn.Module):
    block_size = module.block_size

    def forward(input: torch.Tensor) -> torch.Tensor:
        return _DequantLinearFunction.apply(
            input, module.weight, module.weight_scale_inv, module.bias, block_size
        )

    return forward


class _DequantLinearFunction(torch.autograd.Function):
    """dequant -> F.linear that does NOT save the dequantized weight for backward.

    A plain F.linear(x, w_deq) makes autograd keep w_deq (bf16, ~2x the fp8
    weights of the whole model) alive until backward for the input gradient.
    With gradient checkpointing disabled that extra resident copy wipes out the
    memory savings of fp8 storage (OOM observed at S=2048, see ANALYSYS.md 8.5).
    Here backward re-dequantizes from fp8 instead and saves nothing: the frozen
    weight needs no grad_weight, grad_input = grad_output @ weight does not
    depend on the input, and save_for_backward(input) would pin every layer's
    activation until its backward (~8 GiB model-wide, see FP8_CHECK.md D1).
    """

    @staticmethod
    def forward(ctx, input, weight_fp8, weight_scale_inv, bias, block_size):
        assert bias is None, "bias is not supported by the fp8 dequant path"
        weight = _ACTIVE_DEQUANT(weight_fp8, weight_scale_inv, block_size, input.dtype)
        out = F.linear(input, weight)
        ctx.fp8_args = (weight_fp8, weight_scale_inv, block_size)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        weight_fp8, weight_scale_inv, block_size = ctx.fp8_args
        assert not ctx.saved_tensors
        weight = _ACTIVE_DEQUANT(weight_fp8, weight_scale_inv, block_size, grad_output.dtype)
        grad_input = grad_output @ weight
        return grad_input, None, None, None, None


def patch_fp8_linear_for_training(module: torch.nn.Module) -> bool:
    """Replace module.forward with dequant -> bf16 F.linear. Idempotent."""
    if getattr(module, PATCHED_ATTR, False):
        return False
    # Fail fast at patch time: the dequant backward returns no bias gradient, so
    # a bias would be silently untrainable. Target models have no bias on the
    # quantized text linears (all bias keys belong to the visual tower).
    if getattr(module, "bias", None) is not None:
        raise ValueError("quantized FP8Linear with bias is not supported by fp8_storage")
    module.forward = _make_dequant_forward(module)
    setattr(module, PATCHED_ATTR, True)
    return True


def apply_fp8_storage(model: torch.nn.Module, verbose: bool = True) -> int:
    """Patch every quantized FP8Linear in the model for bf16-compute training.

    FP8Linear modules holding an unquantized weight (element_size > 1, e.g. if
    the checkpoint kept them in bf16) are left alone: their own forward already
    falls back to a plain F.linear and is autograd-friendly.

    Also freezes fp8 parameters (weights and block scales): HF leaves params
    trainable after from_pretrained (the fp8 quantizer is inference-only), and
    fp8 weights are storage-only by design.

    Returns the number of patched modules.
    """
    patched = 0
    for module in model.modules():
        if _is_fp8_quantized_linear(module):
            if patch_fp8_linear_for_training(module):
                patched += 1
            module.weight.requires_grad = False
            module.weight_scale_inv.requires_grad = False
    if verbose:
        total_fp8 = sum(1 for m in model.modules() if _is_fp8_quantized_linear(m))
        print(f"[fp8-storage] patched {patched} FP8Linear modules ({total_fp8} quantized total)")
    return patched


def verify_fp8_storage(model: torch.nn.Module) -> Dict[str, float]:
    """Fail-fast checks after loading (and again after PEFT wrapping).

    Raises on:
    - any quantized FP8Linear whose forward was not patched (backward would die
      inside a Triton kernel without autograd);
    - any fp8 parameter with requires_grad=True (fp8 is storage-only).

    Returns a summary dict with parameter bytes per dtype group.
    """
    errors = []
    for name, module in model.named_modules():
        if _is_fp8_quantized_linear(module) and not getattr(module, PATCHED_ATTR, False):
            errors.append(f"unpatched quantized FP8Linear: {name}")
    for name, param in model.named_parameters():
        if param.dtype in FP8_DTYPES and param.requires_grad:
            errors.append(f"trainable fp8 parameter: {name}")
    if errors:
        raise RuntimeError(
            "fp8_storage verification failed:\n  " + "\n  ".join(errors[:20])
        )

    bytes_by_group: Dict[str, int] = {"fp8_frozen": 0, "bf16_frozen": 0, "trainable": 0}
    for param in model.parameters():
        nbytes = param.numel() * param.element_size()
        if param.requires_grad:
            bytes_by_group["trainable"] += nbytes
        elif param.dtype in FP8_DTYPES:
            bytes_by_group["fp8_frozen"] += nbytes
        else:
            bytes_by_group["bf16_frozen"] += nbytes
    return {k: v / 2**30 for k, v in bytes_by_group.items()}
