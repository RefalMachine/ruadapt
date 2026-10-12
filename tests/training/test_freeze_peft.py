import pytest
import torch

from ruadapt.training.core.freeze import register_embed_freeze_hook, protect_frozen_optimizer_rows
from tests.training.test_checkpoint_resume import make_model, make_trainer


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("peft", [False, True])
def test_old_rows_stay_fixed_after_optimizer_steps(tied, peft):
    model = make_model(tied=tied, save_embeddings=True)
    if not peft:
        model = model.merge_and_unload()
    register_embed_freeze_hook(model, 8)
    parameters = list({id(m.weight): m.weight for m in
                       (model.get_input_embeddings(), model.get_output_embeddings())}.values())
    if tied:
        assert len(parameters) == 1
    before = [p.detach().clone() for p in parameters]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01, weight_decay=0)
    protect_frozen_optimizer_rows(model, optimizer)
    inputs = torch.tensor([[1, 2, 9, 10]])
    for step in range(3):
        optimizer.zero_grad()
        model(input_ids=inputs, labels=inputs).loss.backward()
        # Simulate optimizer state restored from a legacy, incorrectly frozen run.
        if step == 1:
            for p in parameters:
                optimizer.state[p]["exp_avg"][:8].fill_(1)
                optimizer.state[p]["exp_avg_sq"][:8].fill_(1)
        optimizer.step()
    for p, initial in zip(parameters, before):
        assert torch.equal(p[:8], initial[:8])
        assert not torch.equal(p[8:], initial[8:])


def test_partial_freeze_resume_and_changed_boundary(tmp_path):
    original = make_model(tied=True, save_embeddings=True)
    register_embed_freeze_hook(original, 8)
    trainer = make_trainer(tmp_path / "full", model=original)
    trainer.train()
    resumed_model = make_model(tied=True, save_embeddings=True)
    register_embed_freeze_hook(resumed_model, 8)
    resumed = make_trainer(tmp_path / "resumed", model=resumed_model)
    checkpoint = str(tmp_path / "full" / "checkpoint-2")
    resumed.train(resume_from_checkpoint=checkpoint)
    torch.testing.assert_close(resumed.model.get_input_embeddings().weight,
                               trainer.model.get_input_embeddings().weight, rtol=0, atol=0)
    changed_model = make_model(tied=True, save_embeddings=True)
    register_embed_freeze_hook(changed_model, 9)
    changed = make_trainer(tmp_path / "changed", model=changed_model)
    with pytest.raises(ValueError, match="freeze configuration"):
        changed.train(resume_from_checkpoint=checkpoint)


def test_decay_rejected_even_if_loaded_later():
    model = make_model(save_embeddings=True)
    register_embed_freeze_hook(model, 8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0)
    protect_frozen_optimizer_rows(model, optimizer)
    optimizer.param_groups[0]["weight_decay"] = 0.1
    with pytest.raises(ValueError, match="weight_decay"):
        optimizer.step()


@pytest.mark.parametrize("limit", [-1, 33, 1.5, True])
def test_invalid_freeze_boundary(limit):
    with pytest.raises(ValueError, match="freeze_idx"):
        register_embed_freeze_hook(make_model(), limit)


def test_frozen_embeddings_are_skipped():
    model = make_model()  # plain LoRA freezes the embeddings entirely
    register_embed_freeze_hook(model, 8)
    assert not hasattr(model.get_input_embeddings().weight, "_ruadapt_freeze_idx")


def test_guard_survives_accelerate_optimizer_wrapper(tmp_path):
    model = make_model(tied=True, save_embeddings=True)
    register_embed_freeze_hook(model, 8)
    trainer = make_trainer(tmp_path, model=model, max_steps=1, save_strategy="no")
    trainer.train()
    wrapped = trainer.optimizer
    assert hasattr(wrapped, "optimizer")
    assert trainer.create_optimizer() is wrapped
