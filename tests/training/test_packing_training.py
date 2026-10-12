"""Factories and Trainer correctness beyond attention-kernel parity."""

import json

import pytest
import torch
from datasets import Dataset
from transformers import TrainingArguments

from ruadapt.training.config.packing import PackingPolicy
from ruadapt.training.core.trainer import UnifiedTrainer
from ruadapt.training.datasets.collators import DocumentCollator
from ruadapt.training.datasets.document_packing import iter_packs
from tests.training.test_document_packing import records, tiny_model


def training_setup(tmp_path, batch, accumulation=1, *, lora=False):
    p = PackingPolicy("cpt", "isolated", 16, "tokens")
    dataset = Dataset.from_list(list(iter_packs(records(), mode=p.mode, max_length=p.max_length)))
    model = tiny_model("qwen3", "sdpa", lora=lora)
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[], disable_tqdm=True,
                             per_device_train_batch_size=batch, per_device_eval_batch_size=batch,
                             gradient_accumulation_steps=accumulation, max_steps=1, save_strategy="no",
                             learning_rate=0.01, lr_scheduler_type="constant", optim="sgd",
                             weight_decay=0, max_grad_norm=0, remove_unused_columns=False)
    trainer = UnifiedTrainer(model=model, args=args, train_dataset=dataset, eval_dataset=dataset,
                             data_collator=DocumentCollator(p, 0), packing_policy=p)
    return trainer


@pytest.mark.parametrize("lora", [False, True])
def test_accumulation_matches_multiple_packs(tmp_path, lora):
    one = training_setup(tmp_path / "one", 1, 2, lora=lora)
    two = training_setup(tmp_path / "two", 2, lora=lora)
    one.train()
    two.train()
    for (name, a), (_, b) in zip(one.model.named_parameters(), two.model.named_parameters()):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=2e-7, msg=name)


def test_eval_is_token_weighted_and_counts_shifted_labels(tmp_path):
    one = training_setup(tmp_path / "one", 1)
    two = training_setup(tmp_path / "two", 2)
    a, b = one.evaluate(), two.evaluate()
    assert a["eval_target_count"] == b["eval_target_count"] == 20
    assert a["eval_loss"] == pytest.approx(b["eval_loss"], abs=1e-6)


def test_empty_targets_and_checkpoint_policy(tmp_path):
    t = training_setup(tmp_path, 1)
    batch = next(iter(t.get_train_dataloader()))
    batch["labels"].fill_(-100)
    count = t._get_num_items_in_batch([batch], torch.device("cpu"))
    loss = t.compute_loss(t.model, batch, num_items_in_batch=count)
    assert torch.isfinite(loss) and loss.item() == 0
    loss.backward()
    t.save_model()
    assert json.loads((tmp_path / "ruadapt_packing.json").read_text())["mode"] == "isolated"
    t.packing_policy = PackingPolicy("cpt", "stream", 16, "tokens")
    with pytest.raises(ValueError, match="packing/loss policy"):
        t._load_from_checkpoint(str(tmp_path))


@pytest.mark.parametrize("backend", ["sdpa", "flash_attention_2"])
@pytest.mark.parametrize("liger", [False, True])
def test_hybrid_trainer_checkpointing(tmp_path, backend, liger):
    if not torch.cuda.is_available():
        pytest.skip("Qwen3.5 FLA requires CUDA")
    if liger:
        pytest.importorskip("liger_kernel")
    p = PackingPolicy("cpt", "isolated", 16, "tokens")
    data = Dataset.from_list(list(iter_packs(records(), mode=p.mode, max_length=p.max_length)))
    model = tiny_model("qwen3_5", backend, "cuda", lora=True)
    model.enable_input_require_grads()
    args = TrainingArguments(output_dir=str(tmp_path), report_to=[], disable_tqdm=True,
                             per_device_train_batch_size=2, per_device_eval_batch_size=2,
                             max_steps=1, save_strategy="no", bf16=True,
                             gradient_checkpointing=True, use_liger_kernel=liger,
                             remove_unused_columns=False)
    t = UnifiedTrainer(model=model, args=args, train_dataset=data, eval_dataset=data,
                       data_collator=DocumentCollator(p, 0), packing_policy=p)
    assert t.train().training_loss > 0
    if liger:
        assert model.get_base_model().forward.__module__.startswith("liger_kernel")
    result = t.evaluate()
    assert result["eval_target_count"] == 20 and result["eval_loss"] > 0


@pytest.mark.parametrize("regime", ["cpt_stream", "sft_none", "cpt_isolated", "sft_isolated"])
def test_factory_train_eval_end_to_end(tmp_path, chat_tokenizer, tokenizer, regime):
    from ruadapt.training.train import parse_config_from_json
    from ruadapt.training.datasets.factory import load_factory
    from ruadapt.training.config.packing import resolve_packing
    task, mode = regime.split("_")
    chat_tokenizer = chat_tokenizer if task == "sft" else tokenizer
    data = tmp_path / "data.json"
    if task == "sft":
        data.write_text(json.dumps([{"messages": [{"role": "user", "content": "hello"},
                                                  {"role": "assistant", "content": "answer"}]}] * 6))
        factory, collator = "sft_factory.SFTDatasetFactory", "sft_factory.SFTCollatorFactory"
    else:
        data.write_text(json.dumps([{"text": "hello answer " * n} for n in [8, 12, 5, 10]]))
        factory, collator = "unified_factory.UnifiedDatasetFactory", "unified_factory.UnifiedCollatorFactory"
    raw = {
        "model": {"model_name_or_path": "unused", "attn_implementation": "sdpa"},
        "data": {"train_file": str(data), "val_file": str(data), "cache_dir": str(tmp_path / "cache"),
                 "preprocessing_num_workers": 1},
        "packing": {"mode": mode, "max_length": 32},
        "training": {"output_dir": str(tmp_path / "train"), "use_cpu": True, "max_steps": 1,
                     "per_device_train_batch_size": 2, "remove_unused_columns": False,
                     "report_to": [], "disable_tqdm": True, "save_strategy": "no"},
        "dataset_factory": "ruadapt.training.datasets." + factory,
        "collator_factory": "ruadapt.training.datasets." + collator,
    }
    if task == "sft":
        raw["sft"] = {"max_tokens_count": 32}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    cfg = parse_config_from_json(str(path))
    df, cf = load_factory(cfg.dataset_factory)(), load_factory(cfg.collator_factory)()
    train, evaluate = df.create_train(chat_tokenizer, cfg), df.create_eval(chat_tokenizer, cfg)
    policy = resolve_packing(cfg)
    trainer = UnifiedTrainer(model=tiny_model("qwen3", "sdpa"), args=cfg.training,
                             train_dataset=train, eval_dataset=evaluate,
                             data_collator=cf.create(chat_tokenizer, cfg),
                             eval_collator=cf.create_eval(chat_tokenizer, cfg), packing_policy=policy)
    assert trainer.train().global_step == 1
    result = trainer.evaluate()
    assert result["eval_target_count"] > 0 and result["eval_loss"] > 0
    if task == "sft":
        assert len(evaluate) == 6
        assert "segment_lengths" not in evaluate.column_names
