"""Sequence limits, dense isolation, variable tails, and multi-file factory integration."""
import json
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from transformers import TrainingArguments

from ruadapt.training.config.packing import PackingPolicy, resolve_packing
from ruadapt.training.config.schema import PackingConfig, SequenceConfig, SFTConfig
from ruadapt.training.datasets.document_packing import iter_packs, iter_sequences, pack_dataset
from ruadapt.training.datasets.collators import DocumentCollator
from ruadapt.training.core.trainer import UnifiedTrainer
from tests.training.test_document_packing import tiny_model


def rows():
    return [{"input_ids": list(range(a, a + n)), "labels": list(range(a, a + n))}
            for a, n in [(10, 6), (30, 4), (50, 6)]]


@pytest.mark.parametrize("overflow,lengths", [("split", [4, 2, 4, 4, 2]), ("drop", [4]), ("truncate", [4, 4, 4])])
def test_sequence_limits(overflow, lengths):
    result = list(iter_sequences(rows(), max_length=4, overflow=overflow))
    assert [len(r["input_ids"]) for r in result] == lengths
    assert all(r["labels"] == r["input_ids"] for r in result)
    with pytest.raises(ValueError, match="Example 0 exceeds"):
        list(iter_sequences(rows(), max_length=4, overflow="error"))


@pytest.mark.parametrize("tail,expected", [("keep", [8, 2]), ("pad", [8, 2]), ("drop", [8])])
def test_dense_tail_and_boundary_targets(tail, expected):
    from ruadapt.training.datasets.document_packing import packing_stats
    result = list(iter_packs(rows()[:2], mode="isolated", strategy="dense", max_length=8, tail=tail))
    assert [len(r["input_ids"]) for r in result] == expected
    assert result[0]["segment_lengths"] == [6, 2]
    assert result[0]["labels"] == [-100, 11, 12, 13, 14, 15, -100, 31]
    if tail != "drop":
        assert result[1]["input_ids"] == [32, 33]
        assert result[1]["labels"] == [-100, 33]
    policy = PackingPolicy("cpt", "isolated", 8, "tokens", "dense", tail)
    batch = DocumentCollator(policy, 0)(result)
    assert batch["input_ids"].numel() == (10 if tail == "keep" else 8 * len(result))
    if tail == "keep":
        stats = packing_stats(result, policy)
        assert stats["fill"] == 1.0
        assert stats["nominal_pack_fill"] == 10 / 16


def test_whole_drop_only_discards_last_incomplete_pack():
    result = list(iter_packs(rows(), mode="isolated", strategy="whole", max_length=8, tail="drop"))
    assert [len(r["input_ids"]) for r in result] == [6, 4]
    dense = list(iter_packs(rows(), mode="isolated", strategy="dense", max_length=8, tail="drop"))
    assert [len(r["input_ids"]) for r in dense] == [8, 8]
    assert sum((r["input_ids"] for r in dense), []) == sum((r["input_ids"] for r in rows()), [])


@pytest.mark.parametrize("task", ["cpt", "sft"])
def test_common_policy_and_eval(task):
    cfg = SimpleNamespace(data=SimpleNamespace(block_size=None), sft=SFTConfig() if task == "sft" else None,
                          packing=PackingConfig(strategy="dense", isolate=True, max_length=8, tail="drop"),
                          sequence=SequenceConfig(max_length=20, overflow="split"))
    policy = resolve_packing(cfg)
    assert (policy.mode, policy.strategy, policy.tail, policy.sequence_max_length) == ("isolated", "dense", "drop", 20)
    ev = resolve_packing(cfg, evaluation=True)
    assert ev.mode == ("none" if task == "sft" else "isolated")
    assert ev.sequence_max_length == 20
    cfg.packing.mode = "stream"
    with pytest.raises(ValueError, match="conflicts"):
        resolve_packing(cfg)


def test_legacy_checkpoint_accepts_only_equivalent_layout():
    saved = dict(task="cpt", mode="isolated", max_length=8, loss_normalization="tokens")
    compatible = PackingPolicy(**saved, strategy="whole", tail="pad", sequence_max_length=8, overflow="split")
    assert compatible.matches_checkpoint(saved)
    assert compatible.matches_checkpoint(compatible.to_dict())
    for changed in [dict(strategy="dense"), dict(tail="keep"), dict(overflow="drop"), dict(sequence_max_length=4)]:
        assert not PackingPolicy(**(compatible.to_dict() | changed)).matches_checkpoint(saved)


def test_cache_separates_sequence_and_packing_policies(tmp_path):
    data = Dataset.from_list(rows())
    policies = [PackingPolicy("cpt", "isolated", 8, "tokens", strategy, tail, 8, overflow)
                for strategy, tail, overflow in [("whole", "pad", "split"), ("dense", "pad", "split"),
                                                  ("dense", "keep", "split"), ("dense", "keep", "drop")]]
    outputs = [pack_dataset(data, p, cache_dir=str(tmp_path)) for p in policies]
    assert len({d._fingerprint for d in outputs}) == len(outputs)


@pytest.mark.parametrize("architecture", ["qwen3", "qwen3_5"])
@pytest.mark.parametrize("backend", ["sdpa", "flash_attention_2", "flash_attention_3"])
def test_dense_keep_attention_matches_separate_fragments(architecture, backend):
    if not torch.cuda.is_available() and (architecture != "qwen3" or backend != "sdpa"):
        pytest.skip("CUDA kernels required")
    if backend == "flash_attention_3" and torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("FA3 requires Hopper")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = PackingPolicy("cpt", "isolated", 8, "tokens", "dense", "keep")
    packs = list(iter_packs(rows()[:2], mode="isolated", strategy="dense", max_length=8))
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in DocumentCollator(policy, 0)(packs).items()}
    model = tiny_model(architecture, backend, device)
    count = batch["labels"][..., 1:].ne(-100).sum()
    out = model(**batch, num_items_in_batch=count)
    expected = 0
    offset = 0
    for pack in packs:
        start = 0
        for n in pack["segment_lengths"]:
            ids = torch.tensor([pack["input_ids"][start:start+n]], device=device)
            labels = torch.tensor([pack["labels"][start:start+n]], device=device)
            ref = model(input_ids=ids, labels=labels, num_items_in_batch=count)
            expected += ref.loss
            torch.testing.assert_close(out.logits[:, offset:offset+n], ref.logits, rtol=.04, atol=.025)
            start += n
            offset += n
    torch.testing.assert_close(out.loss, expected, rtol=.02, atol=.003)
    out.loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("liger", [False, True])
def test_variable_tail_eval_and_all_masked_fragment(tmp_path, batch_size, liger):
    if liger and not torch.cuda.is_available():
        pytest.skip("Liger requires CUDA")
    if liger:
        pytest.importorskip("liger_kernel")
    device = "cuda" if liger else "cpu"
    policy = PackingPolicy("sft", "isolated", 8, "tokens", "dense", "keep")
    source = rows()[:2]
    source[0]["labels"] = [-100] * 6
    packs = list(iter_packs(source, mode="isolated", strategy="dense", max_length=8))
    data = Dataset.from_list(packs)
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=not liger, use_liger_kernel=liger,
                             bf16=liger, report_to=[], remove_unused_columns=False,
                             per_device_eval_batch_size=batch_size, per_device_train_batch_size=batch_size,
                             max_steps=1, save_strategy="no")
    trainer = UnifiedTrainer(model=tiny_model("qwen3", "sdpa", device), args=args, train_dataset=data,
                             eval_dataset=data, data_collator=DocumentCollator(policy, 0), packing_policy=policy)
    result = trainer.evaluate()
    assert result["eval_target_count"] == 2
    assert result["eval_loss"] > 0
    batch = DocumentCollator(policy, 0)(packs)
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    batch["labels"].fill_(-100)
    loss = trainer.compute_loss(trainer.model, batch)
    assert loss.item() == 0
    loss.backward()


@pytest.mark.parametrize("task", ["cpt", "sft"])
@pytest.mark.parametrize("overflow", ["split", "drop", "truncate", "error"])
def test_factories_multifile_and_long_examples(tmp_path, tokenizer, chat_tokenizer, task, overflow):
    from ruadapt.training.train import parse_config_from_json
    from ruadapt.training.datasets.factory import load_factory
    tok = tokenizer if task == "cpt" else chat_tokenizer
    short = {"text": "hello world"} if task == "cpt" else {"messages": [
        {"role": "user", "content": "hello"}, {"role": "assistant", "content": "answer"}]}
    long = {"text": "hello world " * 100} if task == "cpt" else {"messages": [
        {"role": "user", "content": "hello\n" * 100}, {"role": "assistant", "content": "answer\n" * 100}]}
    for index, record in enumerate([short, long]):
        (tmp_path / f"part-{index}.jsonl").write_text(json.dumps(record) + "\n")
    module = "unified_factory.Unified" if task == "cpt" else "sft_factory.SFT"
    raw = dict(model=dict(model_name_or_path="unused"),
               data=dict(train_file=str(tmp_path / 'part-*.jsonl'), val_file=[str(tmp_path / 'part-0.jsonl')],
                         cache_dir=str(tmp_path / 'cache'), preprocessing_num_workers=1),
               sequence=dict(max_length=32, overflow=overflow),
               packing=dict(strategy="dense", isolate=True, max_length=16, tail="keep"),
               training=dict(output_dir=str(tmp_path / 'out'), use_cpu=True, report_to=[], remove_unused_columns=False),
               dataset_factory="ruadapt.training.datasets." + module + "DatasetFactory",
               collator_factory="ruadapt.training.datasets." + module + "CollatorFactory")
    if task == "sft":
        raw["sft"] = {}
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(raw))
    cfg = parse_config_from_json(str(path))
    factory = load_factory(cfg.dataset_factory)()
    if overflow == "error":
        with pytest.raises(ValueError, match="Packing failed"):
            factory.create_train(tok, cfg)
        return
    train = factory.create_train(tok, cfg)
    total = sum(len(r['input_ids']) for r in train)
    assert total > 0
    if overflow == "split":
        assert total > 100
        if task == "sft":
            assert any(all(y == -100 for y in r['labels']) for r in train)
    elif overflow == "drop":
        assert total < 32
    else:
        assert 32 < total < 64
    evaluate = factory.create_eval(tok, cfg)
    assert len(evaluate) > 0
    if task == "sft":
        assert len(evaluate) == 1
