"""Throughput/memory benchmark for Qwen3.5-27B SFT setup.

Reproduces the production training stack (LoRA r=128 + gradient checkpointing
+ liger + FA3) on synthetic batches and measures tok/s and peak memory for
candidate packing configurations.

Not a pytest test: run manually on a free GPU, e.g.

    CUDA_VISIBLE_DEVICES=4 python tests/perf/bench_qwen35.py

Scenario rationale (see ANALYSYS.md):
    A. B=2, S=448 + attention_mask — current training regime (calibration:
       expected ~550 tok/s per GPU, matching trainer_state metrics).
    B. B=1, S=2048 — packing baseline (chunk 2048).
    C. B=1, S=2048 packed (2 docs) — production packing format: position_ids
       reset + seq_idx(int32) + cu_seq_lens_q, no attention_mask.
    D. B=1, S=4096 — larger chunk option.
    E. B=2, S=2048 — B>1 option.
    F. B=4, S=2048 — OOM probe of the known bs=4 limit (runs last, guarded).
"""

import os
import time

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from peft import LoraConfig, get_peft_model
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

MODEL = "Qwen/Qwen3.5-27B"
N_PARAMS = 27e9
VOCAB = 248320

LORA_TARGETS = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_model():
    print("loading...", flush=True)
    t0 = time.perf_counter()
    model = Qwen3_5ForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_3",
        trust_remote_code=True,
    )
    print(f"loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    model = get_peft_model(
        model,
        LoraConfig(r=128, lora_alpha=128, lora_dropout=0.0, target_modules=LORA_TARGETS),
    )
    model.print_trainable_parameters()

    from transformers.integrations.liger import apply_liger_kernel

    apply_liger_kernel(model, None)
    base = model.get_base_model()
    print("liger fwd:", base.forward.__qualname__, flush=True)
    print("liger mlp[0]:", base.model.layers[0].mlp._get_name(), flush=True)

    model = model.to("cuda")
    model.train()
    print(f"on device, allocated: {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    return model


def mk_inputs(B, S, packed=False, docs=2):
    ids = torch.randint(0, VOCAB - 1000, (B, S), device="cuda")
    labels = ids.clone()
    pos = torch.arange(S, device="cuda").unsqueeze(0).expand(B, -1).contiguous()
    if not packed:
        return dict(
            input_ids=ids,
            labels=labels,
            attention_mask=torch.ones(B, S, dtype=torch.long, device="cuda"),
            position_ids=pos,
        )
    assert B == 1 and S % docs == 0, "packed bench supports B=1 with S divisible by docs"
    length = S // docs
    pos_parts, sid, cu = [], [], [0]
    acc, doc = 0, 0
    for _ in range(docs):
        pos_parts.append(torch.arange(length))
        sid.extend([doc] * length)
        acc += length
        cu.append(acc)
        doc += 1
    return dict(
        input_ids=ids,
        labels=labels,
        position_ids=torch.cat(pos_parts).unsqueeze(0).contiguous().cuda(),
        seq_idx=torch.tensor([sid], device="cuda", dtype=torch.int32),
        cu_seq_lens_q=torch.tensor(cu, device="cuda", dtype=torch.int32),
    )


def bench(model, name, B, S, packed=False, docs=2, iters=5, warmup=2):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    inp = mk_inputs(B, S, packed, docs)
    try:
        for _ in range(warmup):
            out = model(**inp)
            out.loss.backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            out = model(**inp)
            out.loss.backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        toks = B * S / dt
        tflops = toks * 6 * N_PARAMS / 1e12
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(
            f"{name}: {dt*1000:.0f} ms/iter | {toks:.0f} tok/s | "
            f"~{tflops:.0f} TFLOPS(6N) | peak {peak:.1f} GiB",
            flush=True,
        )
    except torch.cuda.OutOfMemoryError:
        print(f"{name}: OOM", flush=True)
        torch.cuda.empty_cache()
    del inp


def main():
    model = build_model()
    bench(model, "A current-shape B2 S448 +mask", 2, 448)
    bench(model, "B B1 S2048 dense", 1, 2048)
    bench(model, "C B1 S2048 packed 2docs", 1, 2048, packed=True)
    bench(model, "D B1 S4096 dense", 1, 4096)
    bench(model, "E B2 S2048 dense", 2, 2048)
    bench(model, "F B4 S2048 dense (OOM probe)", 4, 2048)
    bench(model, "G B1 S4096 packed 20docs (prod shape)", 1, 4096, packed=True, docs=20)
    print("done", flush=True)


if __name__ == "__main__":
    main()
