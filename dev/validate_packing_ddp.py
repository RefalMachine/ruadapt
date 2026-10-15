"""Two-rank CPU/Gloo packing, token-weighted eval, checkpoint/resume regression.

PYTHONPATH=. python -m torch.distributed.run --standalone --nproc_per_node=2 \
    dev/validate_packing_ddp.py --output /path/to/fresh/output
"""
import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import TrainingArguments
from safetensors.torch import load_file
from torch.utils.data import SequentialSampler

from ruadapt.training.config.packing import PackingPolicy
from ruadapt.training.core.trainer import UnifiedTrainer
from ruadapt.training.core.distributed import main_process_first
from ruadapt.training.datasets.collators import DocumentCollator
from ruadapt.training.datasets.document_packing import pack_dataset
from ruadapt.training.datasets.unified_factory import ensure_packed
from ruadapt.training.datasets.utils import ensure_tokenized
from tests.training.test_document_packing import tiny_model


def tokenize(batch):
    return {"input_ids": [list(range(3, 3 + n)) for n in batch["length"]]}


def documents(batch, indices):
    return {"input_ids": batch["input_ids"], "labels": batch["input_ids"]}


def main():
    os.environ["ACCELERATE_TORCH_DEVICE"] = "cpu"
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    root = Path(parser.parse_args().output)
    for regime in ["cpt_stream", "sft_none", "cpt_isolated", "sft_isolated", "cpt_dense_keep", "sft_dense_keep"]:
        task, mode, *_ = regime.split("_")
        policy = (PackingPolicy(task, "isolated", 16, "tokens", "dense", "keep", 16, "split")
                  if mode == "dense" else PackingPolicy(task, mode, 16, "tokens"))
        args = TrainingArguments(output_dir=str(root / regime), use_cpu=True, report_to=[],
                                 disable_tqdm=True, per_device_train_batch_size=2,
                                 per_device_eval_batch_size=2, max_steps=2,
                                 save_strategy="steps", save_steps=1, learning_rate=0.01,
                                 optim="sgd", lr_scheduler_type="constant", weight_decay=0,
                                 max_grad_norm=0, remove_unused_columns=False,
                                 ddp_find_unused_parameters=False)
        def prepare():
            raw = Dataset.from_dict({"length": [9, 10, 11, 12, 13, 14] if mode == "dense" else [9, 10, 11, 12, 13]})
            tokenized = ensure_tokenized(raw, tokenize, 1, False, args.process_index == 0)
            docs = ensure_packed(tokenized, documents, 1, False, args.process_index == 0)
            train = pack_dataset(docs, policy, cache_dir=str(root / "cache"))
            evaluate = pack_dataset(docs, policy.for_eval(), cache_dir=str(root / "cache"))
            return train, evaluate
        train, evaluate = main_process_first(prepare)
        def make():
            trainer = UnifiedTrainer(model=tiny_model("qwen3", "sdpa"), args=args,
                                     train_dataset=train, eval_dataset=evaluate,
                                     data_collator=DocumentCollator(policy, 0),
                                     eval_collator=DocumentCollator(policy.for_eval(), 0),
                                     packing_policy=policy)
            trainer._get_train_sampler = lambda train_dataset=None: SequentialSampler(train)
            return trainer
        t = make()
        expected_sum, expected_count = 0.0, 0
        with torch.no_grad():
            for row in evaluate:
                b = DocumentCollator(policy.for_eval(), 0)([row])
                count = b["labels"][..., 1:].ne(-100).sum().item()
                expected_sum += t.model(**b, num_items_in_batch=1).loss.item()
                expected_count += count
        metrics = t.evaluate()
        assert metrics["eval_target_count"] == expected_count, metrics
        assert abs(metrics["eval_loss"] - expected_sum / expected_count) < 2e-6, metrics
        # The first global DDP batch is rows 0..3. Compare its SGD update to
        # one forward over that same global data with the global denominator.
        reference = tiny_model("qwen3", "sdpa")
        global_batch = DocumentCollator(policy, 0)([train[i] for i in range(4)])
        global_count = global_batch["labels"][..., 1:].ne(-100).sum()
        reference(**global_batch, num_items_in_batch=global_count).loss.backward()
        torch.optim.SGD(reference.parameters(), lr=0.01).step()
        t.train()
        expected = {n: p.detach().clone() for n, p in t.model.named_parameters()}
        dist.barrier()
        first_state = load_file(str(root / regime / "checkpoint-1" / "model.safetensors"))
        for name, parameter in reference.named_parameters():
            torch.testing.assert_close(first_state[name], parameter, atol=2e-7, rtol=1e-5)
        resumed = make()
        resumed.train(resume_from_checkpoint=str(root / regime / "checkpoint-1"))
        for name, p in resumed.model.named_parameters():
            torch.testing.assert_close(p, expected[name], atol=0, rtol=0)
        dist.barrier()
        if args.process_index == 0:
            print(f"PASS {regime}: world=2, global SGD parity, eval count={expected_count}, exact resume")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
