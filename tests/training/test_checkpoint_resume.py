"""Real tiny-model training, checkpoint rotation and resume, entirely on CPU."""

import shutil

import torch
from peft import PeftModel, get_peft_model_state_dict
from transformers import LlamaConfig, LlamaForCausalLM, TrainerCallback

from ruadapt.training.config.schema import LoRAConfig, TrainingConfig
from ruadapt.training.core.model import apply_lora_with_tied_embeddings
from ruadapt.training.core.trainer import UnifiedTrainer


def make_model(tied=False, save_embeddings=False):
    torch.manual_seed(7)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, tie_word_embeddings=tied,
        attention_dropout=0.0,
    ))
    model.config._attn_implementation = "eager"
    return apply_lora_with_tied_embeddings(model, LoRAConfig(
        peft=True, r=2, lora_alpha=2, target_modules=["q_proj", "v_proj"],
        modules_to_save=["embed_tokens", "lm_head"] if save_embeddings else None,
    ), verbose=False)


def make_trainer(path, model=None, **overrides):
    args = dict(
        output_dir=str(path), use_cpu=True, report_to=[], max_steps=4,
        per_device_train_batch_size=2, learning_rate=0.01, lr_scheduler_type="linear",
        save_steps=2, save_total_limit=2, logging_strategy="no", disable_tqdm=True,
        seed=123, data_seed=123,
    )
    args.update(overrides)
    data = [{"input_ids": [1, 2, 3 + i, 12], "labels": [1, 2, 3 + i, 12]} for i in range(8)]
    return UnifiedTrainer(model=make_model() if model is None else model,
                          args=TrainingConfig(**args), train_dataset=data)


def adapter_state(model):
    return {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}


def test_resume_matches_uninterrupted_training(tmp_path):
    full = make_trainer(tmp_path / "full")
    full.train()
    checkpoint = tmp_path / "full" / "checkpoint-2"
    for name in ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt",
                 "scheduler.pt", "trainer_state.json", "rng_state.pth"):
        assert (checkpoint / name).is_file(), name
    assert not (checkpoint / "lora_adapter").exists()
    resumed = make_trainer(tmp_path / "resumed")
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == full.state.global_step == 4
    for key, value in adapter_state(full.model).items():
        torch.testing.assert_close(adapter_state(resumed.model)[key], value, rtol=0, atol=0)
    assert resumed.lr_scheduler.state_dict() == full.lr_scheduler.state_dict()
    assert set(resumed.model.peft_config) == {"default"}


def test_best_checkpoint_is_not_overwritten_and_rotation_works(tmp_path):
    class KeepFirst(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if state.global_step == 2:
                state.best_model_checkpoint = str(tmp_path / "checkpoint-2")
                state.best_global_step = 2
                self.saved = (tmp_path / "checkpoint-2" / "adapter_model.safetensors").read_bytes()

    trainer = make_trainer(tmp_path, max_steps=6)
    callback = KeepFirst()
    trainer.add_callback(callback)
    trainer.train()
    assert (tmp_path / "checkpoint-2" / "adapter_model.safetensors").read_bytes() == callback.saved
    assert not (tmp_path / "checkpoint-4").exists()
    assert (tmp_path / "checkpoint-6").exists()
    assert callback.saved != (tmp_path / "checkpoint-6" / "adapter_model.safetensors").read_bytes()


def test_legacy_duplicate_adapter_does_not_hijack_resume(tmp_path):
    trained = make_trainer(tmp_path / "old")
    trained.train()
    root = tmp_path / "old" / "checkpoint-2"
    duplicate = root / "lora_adapter"
    duplicate.mkdir()
    # Deliberately put DIFFERENT weights in the legacy alias.
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy2(tmp_path / "old" / "checkpoint-4" / name, duplicate / name)
    resumed = make_trainer(tmp_path / "new")
    resumed._load_from_checkpoint(str(root))
    expected = make_model()
    expected.load_adapter(str(root), "default", is_trainable=True)
    assert set(resumed.model.peft_config) == {"default"}
    for key, value in adapter_state(expected).items():
        torch.testing.assert_close(adapter_state(resumed.model)[key], value)


def test_final_export_is_loadable_and_matches_root(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.train()
    trainer.save_model()
    trainer.export_final_adapter()
    destination = tmp_path / "final_lora_adapter"
    assert (destination / "adapter_model.safetensors").read_bytes() == (tmp_path / "adapter_model.safetensors").read_bytes()
    base = LlamaForCausalLM(make_model().config)
    loaded = PeftModel.from_pretrained(base, destination)
    for key, value in adapter_state(trainer.model).items():
        torch.testing.assert_close(adapter_state(loaded)[key], value)


def test_nonwriter_export_does_not_touch_files(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path)
    monkeypatch.setattr(type(trainer.args), "should_save", property(lambda self: False))
    trainer.export_final_adapter()
    assert not (tmp_path / "final_lora_adapter").exists()
