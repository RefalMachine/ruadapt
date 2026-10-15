"""CPU/Gloo regression: two-rank adapter save and exact resume.

Run in the development environment:
    PYTHONPATH=. python -m torch.distributed.run --standalone --nproc_per_node=2 \
        dev/validate_training_ddp.py --output /path/to/fresh/output
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from tests.training.test_checkpoint_resume import adapter_state, make_trainer


def main():
    # Accelerate in the production image otherwise selects cpu:0; torch.load
    # cannot use that indexed CPU device as map_location during DDP resume.
    os.environ["ACCELERATE_TORCH_DEVICE"] = "cpu"
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    trainer = make_trainer(root / "full", ddp_find_unused_parameters=False)
    trainer.train()
    expected = adapter_state(trainer.model)
    dist.barrier()
    resumed = make_trainer(root / "resume", ddp_find_unused_parameters=False)
    resumed.train(resume_from_checkpoint=str(root / "full" / "checkpoint-2"))
    for name, value in adapter_state(resumed.model).items():
        torch.testing.assert_close(value, expected[name], atol=0, rtol=0)
    resumed.save_model()
    resumed.export_final_adapter()
    dist.barrier()
    if dist.get_rank() == 0:
        checkpoint = root / "full" / "checkpoint-2"
        for rank in range(dist.get_world_size()):
            assert (checkpoint / f"rng_state_{rank}.pth").is_file()
        assert not (checkpoint / "lora_adapter").exists()
        assert (root / "resume" / "final_lora_adapter" / "adapter_model.safetensors").is_file()
        result = {"world_size": dist.get_world_size(), "backend": dist.get_backend(),
                  "resume_exact": True, "steps": resumed.state.global_step}
        (root / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
