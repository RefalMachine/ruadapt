"""Numerical parity: fp8_storage training path vs transformers' Fp8Dequantize.

Loads the FP8 checkpoint through the production path (fp8 storage + patched
dequant forwards + LoRA), then converts the same module tree in-place to bf16
with transformers' own Fp8Dequantize op (the reference math) and compares
logits with identical adapter weights. Any divergence is the fp8-storage
path's own error: expected <=0.5% (bf16 scale rounding); tolerance 3%.

Note on load paths: FineGrainedFP8Config(dequantize=True) loads correct bf16
weights through Qwen3_5ForConditionalGeneration, but silently produces
unscaled weights through Qwen3_5ForCausalLM (checkpoint keys carry the
`model.language_model.` prefix, which the text-only converter chain does not
pair with the scale tensors — they surface as UNEXPECTED in the load report).
The in-place reference below is immune to that trap.

Heavy test (27B, ~3 min): off by default.
    RUN_FP8_PARITY=1 pytest tests/training/test_fp8_parity.py -v
"""

import os

import pytest
import torch
import torch.nn as nn

MODEL = "Qwen/Qwen3.5-27B-FP8"

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(
        os.environ.get("RUN_FP8_PARITY") != "1", reason="heavy; set RUN_FP8_PARITY=1"
    ),
]


def _randomize_lora_b(model: nn.Module, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    for name, param in model.named_parameters():
        if "lora_B" in name:
            with torch.no_grad():
                param.copy_(torch.randn(param.shape, generator=g) * 0.01)


def _logits(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        out = model(input_ids=input_ids)
    return out.logits.to("cpu", torch.float32)


def test_fp8_storage_matches_hf_dequant():
    import peft
    from types import SimpleNamespace

    from transformers.integrations.finegrained_fp8 import Fp8Dequantize

    from ruadapt.training.config.schema import ModelConfig
    from ruadapt.training.core.fp8 import PATCHED_ATTR, _is_fp8_quantized_linear
    from ruadapt.training.core.model import load_model_and_tokenizer

    torch.manual_seed(1337)
    input_ids = torch.randint(0, 248_000, (1, 256)).to("cuda")
    lora_cfg = dict(r=16, lora_alpha=16, target_modules=["q_proj", "v_proj", "down_proj"])

    # Production path: fp8 storage, patched dequant forwards, same LoRA weights
    cfg = ModelConfig(
        model_name_or_path=MODEL,
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
        trust_remote_code=True,
        text_only=True,
        fp8_storage=True,
        fp8_compile_dequant=False,
    )
    model, _, _ = load_model_and_tokenizer(cfg)
    model = model.to("cuda")
    torch.manual_seed(7)
    model = peft.get_peft_model(model, peft.LoraConfig(**lora_cfg))
    _randomize_lora_b(model, seed=11)

    logits_fp8 = _logits(model, input_ids)

    # Convert the same module tree in-place to bf16 with HF's reference op and
    # restore the native (bf16) forwards.
    deq = Fp8Dequantize(
        SimpleNamespace(quantization_config={"weight_block_size": [128, 128]})
    )
    n_conv = 0
    for mod in model.get_base_model().modules():
        if not _is_fp8_quantized_linear(mod):
            continue
        out = deq.convert(
            {
                "w": [mod.weight.detach().cpu()],
                "w_scale_inv": [mod.weight_scale_inv.detach().cpu()],
            }
        )
        ref = out["w"][0] if isinstance(out["w"], list) else out["w"]
        mod.weight = nn.Parameter(
            ref.to(torch.bfloat16).to(mod.weight.device), requires_grad=False
        )
        del mod.forward  # instance override off -> class forward, bf16 branch
        delattr(mod, PATCHED_ATTR)
        n_conv += 1
    assert n_conv == 400  # 48x3 linear-attn + 64x3 mlp + 16x4 full-attn

    logits_ref = _logits(model, input_ids)

    rel = (logits_fp8 - logits_ref).norm() / logits_ref.norm()
    assert rel < 0.03, f"fp8-storage path diverges from HF dequant: rel={rel:.4f}"
