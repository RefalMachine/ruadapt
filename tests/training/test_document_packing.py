"""Shared packing semantics, real causal loss, and model/backend parity."""

import copy
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset

from ruadapt.training.config.packing import PackingPolicy, resolve_packing, validate_legacy_packing_fields
from ruadapt.training.config.schema import PackingConfig, SFTConfig
from ruadapt.training.datasets.document_packing import iter_packs, pack_dataset
from ruadapt.training.datasets.collators import DocumentCollator, get_pad_token_id
from ruadapt.training.core.packing_attention import install_packing_attention


def records(task="cpt"):
    result = []
    for start, n in [(3, 7), (15, 11), (30, 5)]:
        ids = list(range(start, start + n))
        labels = ids.copy()
        if task == "sft":
            labels[:3] = [-100] * 3
        result.append({"input_ids": ids, "labels": labels})
    return result


def test_stream_keeps_context_and_tail():
    rows = records()
    packs = list(iter_packs(rows, mode="stream", max_length=8))
    assert [len(p["input_ids"]) for p in packs] == [8, 8, 7]
    assert sum((p["input_ids"] for p in packs), []) == sum((r["input_ids"] for r in rows), [])
    assert sum((p["labels"] for p in packs), []) == sum((r["labels"] for r in rows), [])


def test_isolated_fragments_and_custom_targets():
    rows = [{"input_ids": list(range(13)), "labels": [40 + i for i in range(13)]}]
    before = copy.deepcopy(rows)
    packs = list(iter_packs(rows, mode="isolated", max_length=8, split_long=True))
    assert [p["segment_lengths"] for p in packs] == [[8], [5]]
    assert packs[1]["labels"] == [-100, 49, 50, 51, 52]
    assert rows == before
    with pytest.raises(ValueError, match="exceeds"):
        list(iter_packs(rows, mode="isolated", max_length=8))


def test_collator_multiple_packs_padding_and_boundaries():
    p = PackingPolicy("cpt", "isolated", 16, "tokens")
    packs = list(iter_packs(records(), mode=p.mode, max_length=p.max_length))
    b = DocumentCollator(p, 0)(packs)
    assert b["input_ids"].shape == (1, 32)
    assert b["cu_seq_lens_q"].tolist() == [0, 7, 16, 27, 32]
    assert b["seq_idx"].tolist()[0] == [0] * 7 + [1] * 9 + [2] * 11 + [3] * 5
    assert b["labels"][0, [0, 7, 16, 27]].tolist() == [-100] * 4
    assert b["max_length_q"] == b["max_length_k"] == 11
    assert get_pad_token_id(SimpleNamespace(pad_token_id=0, eos_token_id=9)) == 0


def test_arrow_packing_and_cache_policy(tmp_path):
    dataset = Dataset.from_list(records())
    stream = pack_dataset(dataset, PackingPolicy("cpt", "stream", 16, "tokens"), cache_dir=str(tmp_path))
    isolated = pack_dataset(dataset, PackingPolicy("cpt", "isolated", 16, "tokens"), cache_dir=str(tmp_path))
    assert stream._fingerprint != isolated._fingerprint
    assert stream[0]["segment_lengths"] == [16]
    assert isolated[0]["segment_lengths"] == [7]
    assert pack_dataset(dataset, PackingPolicy("cpt", "isolated", 16, "tokens"), cache_dir=str(tmp_path))[:] == isolated[:]


def test_document_substitution_independent_of_workers(tokenizer, tmp_path):
    from ruadapt.training.datasets.packing import make_pack_fn
    fn = make_pack_fn(tokenizer, max_length=16, natural_boundaries=False, fragment_ratio=1,
                      p_split=0.7, seed=17, newline_ids=set(), substitution_method="random",
                      return_documents=True)
    ids = tokenizer("Hello world. Another document with some content.", add_special_tokens=False)["input_ids"]
    dataset = Dataset.from_dict({"input_ids": [ids] * 7})
    one = dataset.map(fn, batched=True, with_indices=True, batch_size=2, num_proc=1)
    two = dataset.map(fn, batched=True, with_indices=True, batch_size=3, num_proc=2)
    assert one[:] == two[:]
    assert len(one) == 7
    assert any(x != y for r in one for x, y in zip(r["input_ids"], r["labels"]))


def test_empty_records_pad_eos_and_stats():
    from ruadapt.training.datasets.document_packing import packing_stats
    policy = PackingPolicy("cpt", "isolated", 8, "tokens")
    rows = [{"input_ids": [], "labels": []}, {"input_ids": [3, 0, 4], "labels": [9, 0, 5]}]
    packs = list(iter_packs(rows, mode=policy.mode, max_length=8))
    batch = DocumentCollator(policy, 0)(packs)
    assert batch["labels"].tolist() == [[-100, 0, 5, -100, -100, -100, -100, -100]]
    stats = packing_stats(packs, policy)
    assert (stats["target_tokens"], stats["target_substitutions"], stats["fill"]) == (2, 1, 3 / 8)


def test_block_mask_sliding_window():
    from ruadapt.training.core.packing_attention import block_causal_mask
    mask = block_causal_mask(torch.tensor([0, 0, 0, 1, 1]), window=2)[0, 0]
    assert mask.tolist() == [[True, False, False, False, False], [True, True, False, False, False],
                             [False, True, True, False, False], [False, False, False, True, False],
                             [False, False, False, True, True]]


def test_resolve_and_legacy_conflicts():
    cfg = SimpleNamespace(sft=SFTConfig(packing=True), data=SimpleNamespace(block_size=None))
    assert resolve_packing(cfg).mode == "isolated"
    assert resolve_packing(cfg, evaluation=True).mode == "none"
    cfg.packing = PackingConfig(mode="isolated")
    assert resolve_packing(cfg).loss_normalization == "tokens"
    with pytest.raises(ValueError, match="conflicts"):
        validate_legacy_packing_fields({"packing": {"mode": "none"}, "sft": {"packing": True}})
    with pytest.raises(ValueError, match="conflicts"):
        validate_legacy_packing_fields({"packing": {"mode": "stream", "max_length": 8}, "data": {"block_size": 16}})
    cfg.sft = None
    cfg.dataset_factory = "ruadapt.training.datasets.sft_factory.SFTDatasetFactory"
    assert resolve_packing(cfg).task == "sft"
    assert resolve_packing(cfg, evaluation=True).mode == "none"


def tiny_model(architecture, backend, device="cpu", lora=False):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    from tests.training.test_sft_packing import _tiny_text_config

    if architecture == "qwen3":
        config = Qwen3Config(vocab_size=1000, hidden_size=128, intermediate_size=256,
                            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                            head_dim=64, attention_dropout=0.0)
        cls = Qwen3ForCausalLM
    else:
        config, cls = _tiny_text_config(), Qwen3_5ForCausalLM
    config._attn_implementation = backend
    config.use_cache = False
    torch.manual_seed(42)
    model = cls(config).to(device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    install_packing_attention(model)
    if lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=4, lora_alpha=4, lora_dropout=0,
                                                target_modules=["q_proj", "v_proj"]))
    return model


@pytest.mark.parametrize("architecture", ["qwen3", "qwen3_5"])
@pytest.mark.parametrize("backend", ["sdpa", "flash_attention_2", "flash_attention_3"])
@pytest.mark.parametrize("regime", ["cpt_stream", "sft_none", "cpt_isolated", "sft_isolated"])
@pytest.mark.parametrize("lora", [False, True])
def test_model_matrix_real_loss_backward(architecture, backend, regime, lora):
    if not torch.cuda.is_available():
        if architecture != "qwen3" or backend != "sdpa":
            pytest.skip("Requires CUDA with FA/FLA kernels")
        device = "cpu"
    else:
        device = "cuda"
    if backend == "flash_attention_3" and torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("FA3 validation requires Hopper hardware")
    task, mode = regime.split("_")
    p = PackingPolicy(task, mode, 16, "tokens")
    rows = records(task)
    packs = list(iter_packs(rows, mode=mode, max_length=16))
    b = DocumentCollator(p, 0)(packs)
    b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
    model = tiny_model(architecture, backend, device, lora)
    model.train()
    count = b["labels"][..., 1:].ne(-100).sum()
    out = model(**b, num_items_in_batch=count)
    logits = out.logits.detach().float()
    out.loss.backward()
    actual_grads = {n: x.grad.detach().float().clone() for n, x in model.named_parameters() if x.grad is not None}
    model.zero_grad(set_to_none=True)

    reference = packs if mode == "stream" else rows
    expected_loss = 0
    expected_logits = []
    for r in reference:
        ids = torch.tensor([r["input_ids"]], device=device)
        labs = torch.tensor([r["labels"]], device=device)
        o = model(input_ids=ids, labels=labs, num_items_in_batch=count)
        expected_loss += o.loss.detach().float()
        expected_logits.append(o.logits.detach().float())
        o.loss.backward()
    torch.testing.assert_close(out.loss.float(), expected_loss, rtol=0.015, atol=0.003)
    numerator = denominator = 0.0
    for n, x in model.named_parameters():
        if n in actual_grads:
            numerator += (actual_grads[n] - x.grad.float()).square().sum().item()
            denominator += x.grad.float().square().sum().item()
    assert (numerator / max(denominator, 1e-20)) ** 0.5 < 0.06

    if mode == "isolated":
        # Use the canonical packs to locate real docs, including SFT's masked prefixes.
        offsets, base = [], 0
        for pack in packs:
            cursor = base
            for length in pack["segment_lengths"]:
                offsets.append(cursor)
                cursor += length
            base += p.max_length
        for offset, ref in zip(offsets, expected_logits):
            torch.testing.assert_close(logits[:, offset:offset + ref.shape[1]], ref, rtol=0.04, atol=0.025)
        changed = {**b, "input_ids": b["input_ids"].clone()}
        changed["input_ids"][:, :7] += 50
        with torch.no_grad():
            changed_logits = model(**changed).logits.float()
        torch.testing.assert_close(changed_logits[:, offsets[1]:], logits[:, offsets[1]:], rtol=0, atol=0)


def test_sdpa_cpu_isolation_and_stream_control():
    model = tiny_model("qwen3", "sdpa")
    p = PackingPolicy("cpt", "isolated", 16, "tokens")
    rows = records()[:2]
    b = DocumentCollator(p, 0)(list(iter_packs(rows, mode="isolated", max_length=16)))
    with torch.no_grad():
        original = model(**b).logits
        changed = {**b, "input_ids": b["input_ids"].clone()}
        changed["input_ids"][:, :7] += 100
        torch.testing.assert_close(model(**changed).logits[:, 16:], original[:, 16:], rtol=0, atol=0)
        ids = torch.tensor([rows[0]["input_ids"] + rows[1]["input_ids"]])
        stream = model(ids).logits
        ids[:, :7] += 100
        assert (model(ids).logits[:, 7:] - stream[:, 7:]).abs().max() > 1e-4
