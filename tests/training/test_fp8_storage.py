"""Tests for FP8 weight storage with bf16 compute (QLoRA-style), core/fp8.py.

All tests are CPU-only and use synthetic tensors / tiny modules.
"""

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from ruadapt.training.core.fp8 import (
    FP8_DTYPES,
    PATCHED_ATTR,
    _is_fp8_quantized_linear,
    apply_fp8_storage,
    dequantize_fp8_block,
    verify_fp8_storage,
)
from ruadapt.training.core.model import (
    _build_fp8_quantization_config,
    _rewrite_modules_to_not_convert,
)

FP8Linear = pytest.importorskip(
    "transformers.integrations.finegrained_fp8", reason="transformers fp8 integration missing"
).FP8Linear
Fp8Quantize = pytest.importorskip(
    "transformers.integrations.finegrained_fp8", reason="transformers fp8 integration missing"
).Fp8Quantize


def _quantize_for_tests(weight: torch.Tensor, block: tuple[int, int]):
    """Quantize with transformers' own Fp8Quantize (the checkpoint format)."""
    from types import SimpleNamespace

    quantizer = SimpleNamespace(quantization_config={"weight_block_size": list(block)})
    out = Fp8Quantize(quantizer).convert({"layer.weight": weight})
    return out["layer.weight"], out["layer.weight_scale_inv"]


def _make_fp8_linear(out_features, in_features, block=(16, 16), seed=0):
    """FP8Linear with block-quantized random weights (as loaded from a checkpoint)."""
    assert out_features % block[0] == 0 and in_features % block[1] == 0, (
        "HF Fp8Quantize passes through shapes not divisible by the block"
    )
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(out_features, in_features, generator=gen, dtype=torch.float32)
    w_q, scale_inv = _quantize_for_tests(w, block)
    mod = FP8Linear(
        in_features, out_features, block_size=block, activation_scheme="dynamic"
    )
    mod.weight = nn.Parameter(w_q, requires_grad=False)
    mod.weight_scale_inv = nn.Parameter(scale_inv, requires_grad=False)
    return mod, w


class TestDequantQuant:
    def test_roundtrip_shapes_and_dtypes(self):
        w = torch.randn(128, 256)
        w_q, scale_inv = _quantize_for_tests(w, (64, 64))
        assert w_q.shape == (128, 256)
        assert w_q.dtype == torch.float8_e4m3fn
        assert scale_inv.shape == (2, 4)
        assert scale_inv.dtype == torch.float32

    def test_dequant_non_divisible_dims(self):
        # Fp8Quantize passes non-divisible shapes through unchanged, but a
        # checkpoint slice can still be non-divisible; dequant must handle it.
        w_q = torch.randn(100, 96).to(torch.float8_e4m3fn)
        scale_inv = torch.rand(2, 2) + 0.5
        back = dequantize_fp8_block(w_q, scale_inv, (64, 64), dtype=torch.float32)
        assert back.shape == (100, 96)
        ref = torch.empty(100, 96, dtype=torch.float32)
        for i in range(2):
            for j in range(2):
                h = min(64, 100 - i * 64)
                wdt = min(64, 96 - j * 64)
                blk = w_q[i * 64 : i * 64 + h, j * 64 : j * 64 + wdt].to(torch.float32)
                ref[i * 64 : i * 64 + h, j * 64 : j * 64 + wdt] = blk * scale_inv[i, j]
        torch.testing.assert_close(back, ref)

    def test_roundtrip_error_bounded(self):
        torch.manual_seed(1)
        w = torch.randn(256, 512) * 0.02
        w_q, scale_inv = _quantize_for_tests(w, (128, 128))
        back = dequantize_fp8_block(w_q, scale_inv, (128, 128), dtype=torch.float32)
        assert (back - w).abs().max() / w.abs().max() < 0.10
        assert (back - w).norm() / w.norm() < 0.03

    def test_dequant_matches_reference(self):
        torch.manual_seed(2)
        w = torch.randn(32, 32)
        w_q, scale_inv = _quantize_for_tests(w, (16, 16))
        ref = torch.empty(32, 32, dtype=torch.float32)
        for i in range(2):
            for j in range(2):
                blk = w_q[i * 16 : (i + 1) * 16, j * 16 : (j + 1) * 16].to(torch.float32)
                ref[i * 16 : (i + 1) * 16, j * 16 : (j + 1) * 16] = blk * scale_inv[i, j]
        out = dequantize_fp8_block(w_q, scale_inv, (16, 16), dtype=torch.float32)
        torch.testing.assert_close(out, ref)

    def test_dequant_per_tensor(self):
        w_q = torch.randn(8, 8).to(torch.float8_e4m3fn)
        scale = torch.tensor(2.5)
        out = dequantize_fp8_block(w_q, scale, None, dtype=torch.float32)
        torch.testing.assert_close(out, w_q.to(torch.float32) * 2.5)

    def test_dequant_does_not_mutate_source_when_dtype_matches(self):
        # D3 regression: to(dtype) returns the same object when dtype matches,
        # and the in-place mul_ would then destroy the source weight (a real
        # hazard for the bf16 merge path).
        w_src = torch.randn(32, 32, dtype=torch.bfloat16)
        before = w_src.clone()
        scale = torch.ones(2, 2, dtype=torch.float32) * 3.0
        out = dequantize_fp8_block(w_src, scale, (16, 16), dtype=torch.bfloat16)
        assert torch.equal(before, w_src), "source weight mutated"
        torch.testing.assert_close(out, before * 3.0)


class TestPatchedForward:
    def test_forward_matches_manual_dequant(self):
        mod, _ = _make_fp8_linear(16, 32, block=(16, 16))
        assert _is_fp8_quantized_linear(mod)
        n = apply_fp8_storage(mod, verbose=False)
        assert n == 1

        x = torch.randn(4, 32)
        y = mod(x)
        w_deq = dequantize_fp8_block(mod.weight, mod.weight_scale_inv, (16, 16), dtype=x.dtype)
        torch.testing.assert_close(y, torch.nn.functional.linear(x, w_deq))

    def test_backward_reaches_input_not_weights(self):
        mod, _ = _make_fp8_linear(16, 32, block=(16, 16))
        apply_fp8_storage(mod, verbose=False)
        x = torch.randn(4, 32, requires_grad=True)
        mod(x).sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert mod.weight.grad is None
        assert not mod.weight.requires_grad

    def test_grad_matches_reference_and_no_dequant_weight_saved(self):
        mod, _ = _make_fp8_linear(16, 32, block=(16, 16))
        apply_fp8_storage(mod, verbose=False)

        x1 = torch.randn(4, 32, requires_grad=True)
        y1 = mod(x1)
        # Nothing saved for backward: no resident bf16 weight copy (re-dequant
        # in backward) and no pinned input either (FP8_CHECK.md, D1).
        assert y1.grad_fn.saved_tensors == ()

        w_deq = dequantize_fp8_block(mod.weight, mod.weight_scale_inv, (16, 16), dtype=x1.dtype)
        x2 = x1.detach().clone().requires_grad_(True)
        y2 = torch.nn.functional.linear(x2, w_deq)
        y1.sum().backward()
        y2.sum().backward()
        torch.testing.assert_close(x1.grad, x2.grad)

    def test_input_not_pinned_until_backward(self):
        # D1 regression: forward must not keep the layer input alive (it used
        # to be save_for_backward'd, pinning ~8 GiB of activations model-wide).
        import gc
        import weakref

        mod, _ = _make_fp8_linear(32, 64, block=(16, 16))
        apply_fp8_storage(mod, verbose=False)
        base = torch.randn(8, 64, requires_grad=True)
        tmp = base * 2.0
        ref = weakref.ref(tmp)
        out = mod(tmp)  # out stays alive until backward, as in training
        del tmp
        gc.collect()
        assert ref() is None, "input pinned after forward"
        out.sum().backward()
        assert base.grad is not None

    def test_patch_rejects_bias(self):
        # D10: bias would be silently untrainable (backward returns None for it).
        mod, _ = _make_fp8_linear(16, 16, block=(16, 16))
        mod.bias = nn.Parameter(torch.zeros(16), requires_grad=False)
        with pytest.raises(ValueError, match="bias"):
            apply_fp8_storage(mod, verbose=False)

    def test_patch_idempotent(self):
        mod, _ = _make_fp8_linear(16, 16, block=(16, 16))
        assert apply_fp8_storage(mod, verbose=False) == 1
        assert apply_fp8_storage(mod, verbose=False) == 0
        assert getattr(mod, PATCHED_ATTR)

    def test_skips_bf16_weight_fp8_linear(self):
        # As if the module was dequantized at load: FP8Linear with bf16 weight.
        mod = FP8Linear(16, 16, block_size=(16, 16), dtype=torch.bfloat16)
        mod.weight = nn.Parameter(torch.randn(16, 16, dtype=torch.bfloat16), requires_grad=False)
        assert not _is_fp8_quantized_linear(mod)
        assert apply_fp8_storage(mod, verbose=False) == 0
        # Original forward (element_size > 1 branch) works and is autograd-friendly.
        x = torch.randn(2, 16, dtype=torch.bfloat16, requires_grad=True)
        mod(x).sum().backward()
        assert x.grad is not None

    def test_verify_raises_on_unpatched(self):
        mod, _ = _make_fp8_linear(16, 16, block=(16, 16))
        with pytest.raises(RuntimeError, match="unpatched quantized FP8Linear"):
            verify_fp8_storage(mod)
        apply_fp8_storage(mod, verbose=False)
        verify_fp8_storage(mod)

    def test_verify_raises_on_trainable_fp8(self):
        mod, _ = _make_fp8_linear(16, 16, block=(16, 16))
        apply_fp8_storage(mod, verbose=False)
        mod.weight.requires_grad = True
        with pytest.raises(RuntimeError, match="trainable fp8 parameter"):
            verify_fp8_storage(mod)

    def test_verify_summary_groups(self):
        mod, _ = _make_fp8_linear(16, 16, block=(16, 16))
        apply_fp8_storage(mod, verbose=False)
        summary = verify_fp8_storage(mod)
        assert summary["fp8_frozen"] > 0
        assert summary["trainable"] == 0


class TestPeftOverPatchedFp8:
    def test_lora_over_fp8_base(self):
        peft = pytest.importorskip("peft")

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj, _ = _make_fp8_linear(16, 16, block=(16, 16))
                self.head = nn.Linear(16, 4)

            def forward(self, x):
                return self.head(self.proj(x))

        model = Tiny()
        apply_fp8_storage(model, verbose=False)

        peft_model = peft.get_peft_model(
            model, peft.LoraConfig(r=4, lora_alpha=4, target_modules=["proj"])
        )

        trainable = {n for n, p in peft_model.named_parameters() if p.requires_grad}
        assert trainable and all("lora_" in n for n in trainable)

        base_lora = peft_model.base_model.model.proj
        base_layer = base_lora.get_base_layer()
        assert getattr(base_layer, PATCHED_ATTR, False), "patch lost under PEFT wrapping"
        assert base_layer.weight.dtype in FP8_DTYPES

        # Adapter dtype ends up fp32 (PEFT upcasts fp8 -> fp32 via cast_adapter_dtype)
        lora_dtypes = {
            base_lora.lora_A["default"].weight.dtype,
            base_lora.lora_B["default"].weight.dtype,
        }
        assert lora_dtypes == {torch.float32}

        w_before = base_layer.weight.detach().clone()
        x = torch.randn(2, 16)
        peft_model(x).sum().backward()
        for n, p in peft_model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and torch.isfinite(p.grad).all(), n
        assert torch.equal(base_layer.weight, w_before)


class TestConfigAndRewrite:
    def test_rewrite_text_only(self):
        names = [
            "lm_head",
            "model.language_model.embed_tokens",
            "model.language_model.layers.0.linear_attn.conv1d",
            "model.language_model.layers.0.linear_attn.in_proj_a",
            "model.visual.blocks.0.attn.qkv",
            "mtp.fc",
        ]
        out = _rewrite_modules_to_not_convert(names, text_only=True)
        assert "lm_head" in out
        assert "model.embed_tokens" in out
        assert "model.layers.0.linear_attn.conv1d" in out
        assert "model.layers.0.linear_attn.in_proj_a" in out
        assert not any("visual" in n or "mtp" in n for n in out)

    def test_rewrite_multimodal_keeps_all(self):
        names = ["lm_head", "model.visual.blocks.0.attn.qkv", "mtp.fc"]
        out = _rewrite_modules_to_not_convert(names, text_only=False)
        assert set(out) == set(names)

    def test_build_quant_config(self):
        hf_config = SimpleNamespace(
            quantization_config=SimpleNamespace(
                quant_method="fp8",
                activation_scheme="dynamic",
                weight_block_size=[128, 128],
                modules_to_not_convert=["lm_head", "model.language_model.embed_tokens"],
            )
        )
        qc = _build_fp8_quantization_config(hf_config, text_only=True)
        assert qc.quant_method == "fp8"
        assert qc.weight_block_size == (128, 128)
        assert "model.embed_tokens" in qc.modules_to_not_convert

    def test_build_quant_config_dict(self):
        # AutoConfig exposes quantization_config as a plain dict
        hf_config = SimpleNamespace(
            quantization_config={
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 128],
                "modules_to_not_convert": ["lm_head", "model.language_model.embed_tokens"],
            }
        )
        qc = _build_fp8_quantization_config(hf_config, text_only=True)
        assert qc.quant_method == "fp8"
        assert qc.weight_block_size == (128, 128)
        assert "model.embed_tokens" in qc.modules_to_not_convert

    def test_build_quant_config_requires_fp8(self):
        with pytest.raises(ValueError, match="quant_method == 'fp8'"):
            _build_fp8_quantization_config(SimpleNamespace(quantization_config=None), True)
        hf_config = SimpleNamespace(
            quantization_config=SimpleNamespace(quant_method="awq")
        )
        with pytest.raises(ValueError, match="quant_method == 'fp8'"):
            _build_fp8_quantization_config(hf_config, True)

    def test_model_config_field_default(self, tmp_path):
        from ruadapt.training.config.schema import ModelConfig
        from ruadapt.training.train import parse_config_from_json

        assert ModelConfig(model_name_or_path="x").fp8_storage is False

        cfg = {
            "model": {"model_name_or_path": "x", "fp8_storage": True},
            "data": {"train_file": "x.jsonl"},
            "training": {"output_dir": "/tmp/o"},
        }
        path = tmp_path / "config.json"
        path.write_text(json.dumps(cfg))
        parsed = parse_config_from_json(str(path))
        assert parsed.model.fp8_storage is True
