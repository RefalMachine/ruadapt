"""Tests for strict SFT document packing (sft_factory.pack_records + PackedSFTCollator).

CPU tests run anywhere. GPU isolation test (packed forward/backward equivalence
with per-document processing on a tiny Qwen3.5) requires CUDA + fla kernels and
is skipped otherwise.
"""

import pytest
import torch

from ruadapt.training.datasets.collators import PackedSFTCollator
from ruadapt.training.datasets.sft_factory import pack_records


# ---------------------------------------------------------------------------
# pack_records: CPU unit tests
# ---------------------------------------------------------------------------

def _sample_batch():
    return {
        "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12], [13]],
        "labels": [[-100, 2, 3], [4, -100, 6, 7], [8, 9, -100, 11, 12], [13]],
    }


def test_pack_chunks_fixed_length_and_greedy():
    out = pack_records(_sample_batch(), pack_chunk_size=8, pad_token_id=0)
    # chunk1: [3] + [4] = 7 tokens + 1 pad; chunk2: [5] + [1] = 6 + 2 pad
    assert len(out["input_ids"]) == 2
    assert all(len(x) == 8 for x in out["input_ids"])
    assert out["input_ids"][0] == [1, 2, 3, 4, 5, 6, 7, 0]
    assert out["input_ids"][1] == [8, 9, 10, 11, 12, 13, 0, 0]


def test_pack_document_boundaries():
    out = pack_records(_sample_batch(), pack_chunk_size=8, pad_token_id=0)
    # position_ids restart at every document incl. pad-tail document
    assert out["position_ids"][0] == [0, 1, 2, 0, 1, 2, 3, 0]
    assert out["position_ids"][1] == [0, 1, 2, 3, 4, 0, 0, 1]
    # seq_idx: unique id per document, pad tail gets its own id
    assert out["seq_idx"][0] == [0, 0, 0, 1, 1, 1, 1, 2]
    assert out["seq_idx"][1] == [0, 0, 0, 0, 0, 1, 2, 2]
    # cu_seqlens cover all segments and end at chunk size
    assert out["cu_seq_lens_q"][0] == [0, 3, 7, 8]
    assert out["cu_seq_lens_q"][1] == [0, 5, 6, 8]


def test_pack_labels_and_pad():
    out = pack_records(_sample_batch(), pack_chunk_size=8, pad_token_id=42)
    assert out["labels"][0] == [-100, 2, 3, 4, -100, 6, 7, -100]
    assert out["input_ids"][0][-1] == 42
    # original label masking inside samples is preserved
    assert out["labels"][1][2] == -100


def test_pack_exact_fill_no_pad_segment():
    batch = {"input_ids": [[1, 2], [3, 4]], "labels": [[1, 2], [3, 4]]}
    out = pack_records(batch, pack_chunk_size=4, pad_token_id=0)
    assert len(out["input_ids"]) == 1
    assert out["input_ids"][0] == [1, 2, 3, 4]
    assert out["cu_seq_lens_q"][0] == [0, 2, 4]
    assert out["seq_idx"][0] == [0, 0, 1, 1]


def test_pack_deterministic():
    out1 = pack_records(_sample_batch(), pack_chunk_size=8, pad_token_id=0)
    out2 = pack_records(_sample_batch(), pack_chunk_size=8, pad_token_id=0)
    assert out1 == out2


def test_pack_drops_too_long_samples():
    batch = {"input_ids": [[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2]], "labels": [[0] * 9, [0, 0]]}
    out = pack_records(batch, pack_chunk_size=4, pad_token_id=0)
    # only the short sample is packed
    assert len(out["input_ids"]) == 1
    assert out["input_ids"][0][:2] == [1, 2]


def test_pack_skips_empty_samples(capsys):
    batch = {"input_ids": [[], [1, 2]], "labels": [[], [1, 2]]}
    out = pack_records(batch, pack_chunk_size=4, pad_token_id=0)
    assert len(out["input_ids"]) == 1
    assert out["input_ids"][0][:2] == [1, 2]


# ---------------------------------------------------------------------------
# PackedSFTCollator
# ---------------------------------------------------------------------------

def test_packed_collator_dtypes_and_keys():
    packed = pack_records(_sample_batch(), pack_chunk_size=8, pad_token_id=0)
    feature = {k: packed[k][0] for k in packed}
    batch = PackedSFTCollator()([feature])
    assert set(batch.keys()) == set(PackedSFTCollator.REQUIRED_KEYS)
    assert "attention_mask" not in batch
    assert batch["input_ids"].dtype == torch.long
    assert batch["labels"].dtype == torch.long
    assert batch["position_ids"].dtype == torch.long
    assert batch["seq_idx"].dtype == torch.int32
    assert batch["cu_seq_lens_q"].dtype == torch.int32
    assert batch["input_ids"].shape == (1, 8)
    # FLA requires 1D cu_seqlens
    assert batch["cu_seq_lens_q"].shape == (4,)


def test_packed_collator_rejects_batch_gt1():
    packed = pack_records(_sample_batch(), pack_chunk_size=8, pad_token_id=0)
    features = [{k: packed[k][i] for k in packed} for i in range(2)]
    with pytest.raises(ValueError):
        PackedSFTCollator()(features)


# ---------------------------------------------------------------------------
# accepts_loss_kwargs landmine (see ANALYSYS.md, section 4 G4)
# ---------------------------------------------------------------------------

def _tiny_text_config():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    return Qwen3_5TextConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        intermediate_size=256,
        layer_types=["full_attention", "linear_attention"],
        linear_conv_kernel_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        tie_word_embeddings=False,
        max_position_embeddings=512,
    )


def test_causal_lm_has_no_accepts_loss_kwargs():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    assert "accepts_loss_kwargs" not in Qwen3_5ForCausalLM.__dict__ and not hasattr(
        Qwen3_5ForCausalLM, "accepts_loss_kwargs"
    )


def test_trainer_loss_kwargs_semantics(tmp_path):
    """Without the attribute Trainer infers accepts=True from **kwargs signature
    (loss normalization would silently change); our fix pins it to False."""
    from transformers import TrainingArguments
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from ruadapt.training.core.trainer import UnifiedTrainer

    model = Qwen3_5ForCausalLM(_tiny_text_config())
    args = TrainingArguments(output_dir=str(tmp_path), report_to=[], use_cpu=True)
    trainer = UnifiedTrainer(model=model, args=args, train_dataset=None, processing_class=None)
    # landmine: **kwargs in forward makes Trainer think loss kwargs are accepted
    assert trainer.model_accepts_loss_kwargs is True

    model.accepts_loss_kwargs = False
    trainer2 = UnifiedTrainer(model=model, args=args, train_dataset=None, processing_class=None)
    assert trainer2.model_accepts_loss_kwargs is False


# ---------------------------------------------------------------------------
# GPU isolation test: packed == per-document (hidden states + LoRA grads)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_packed_isolation_hidden_states_and_lora_grads():
    from peft import LoraConfig, get_peft_model
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(_tiny_text_config()).to("cuda", torch.bfloat16)
    model = get_peft_model(
        model, LoraConfig(r=4, lora_alpha=4, lora_dropout=0.0, target_modules=["q_proj", "v_proj"])
    )
    model.train()

    n_docs, doc_len = 20, 31
    docs = [torch.randint(0, 900, (1, doc_len), device="cuda") for _ in range(n_docs)]
    chunk_size = n_docs * doc_len + 20  # +20 pad tokens as separate tail segment

    # Build the packed chunk through the real pipeline (pack_records + collator)
    batch_in = {
        "input_ids": [d[0].tolist() for d in docs],
        "labels": [d[0].tolist() for d in docs],
    }
    packed = pack_records(batch_in, pack_chunk_size=chunk_size, pad_token_id=0)
    assert len(packed["input_ids"]) == 1
    features = [{k: packed[k][0] for k in packed}]
    inputs = PackedSFTCollator()(features)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    def lora_grads(loss):
        model.zero_grad(set_to_none=True)
        loss.backward()
        grads = []
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                grads.append((n, p.grad.detach().float().clone()))
        return dict(grads)

    # --- packed forward with isolation kwargs ---
    out_packed = model(output_hidden_states=True, use_cache=False, **inputs)
    h_packed = out_packed.hidden_states[-1].float()
    # proxy loss over real document tokens only (pad tail is excluded, as labels=-100 excludes it in training)
    g_packed = lora_grads(out_packed.hidden_states[-1][:, : n_docs * doc_len].sum())

    # --- per-document forwards ---
    h_docs = []
    model.zero_grad(set_to_none=True)
    total = None
    for d in docs:
        pos = torch.arange(doc_len, device="cuda").unsqueeze(0)
        o = model(d, position_ids=pos, use_cache=False, output_hidden_states=True)
        h_docs.append(o.hidden_states[-1].float())
        total = o.hidden_states[-1].sum() if total is None else total + o.hidden_states[-1].sum()
    g_docs = lora_grads(total)

    # hidden states: packed prefix must equal concatenated per-doc outputs.
    # Tolerance is relative: FLA chunk boundaries differ between the packed run
    # and 20 separate runs, giving bf16 rounding noise (~0.7% measured), whereas
    # a real leak grows with document index and is far larger.
    h_cat = torch.cat(h_docs, dim=1)
    diff_h = (h_packed[:, : n_docs * doc_len] - h_cat).abs()
    scale = h_cat.abs().max().item()
    rel_diff_h = diff_h.max().item() / scale
    assert rel_diff_h < 2e-2, f"packed hidden states leak across documents: rel={rel_diff_h}"
    # leak must not grow with document index (real state leakage would)
    per_doc = [diff_h[:, i * doc_len:(i + 1) * doc_len].max().item() for i in range(n_docs)]
    assert per_doc[-1] <= max(per_doc) and per_doc[-1] / scale < 2e-2

    # LoRA gradients must match as well (GatedDeltaNet state leakage shows up here)
    max_rel = 0.0
    for n, gp in g_packed.items():
        gd = g_docs[n]
        denom = gd.norm().item() + 1e-12
        max_rel = max(max_rel, ((gp - gd).norm() / denom).item())
    assert max_rel < 5e-2, f"packed LoRA grads differ from per-document: rel={max_rel}"

    # --- negative control: without seq_idx/cu_seq_lens_q isolation is lost ---
    with torch.no_grad():
        inputs_leak = {k: v for k, v in inputs.items() if k not in ("seq_idx", "cu_seq_lens_q")}
        out_leak = model(output_hidden_states=True, use_cache=False, **inputs_leak)
        h_leak = out_leak.hidden_states[-1].float()
    leak_diff = (h_leak[:, : n_docs * doc_len] - h_cat).abs().max().item()
    assert leak_diff > max(1e-3, 3 * diff_h.max().item()), (
        "negative control failed: GatedDeltaNet leak not detected without kwargs"
    )
