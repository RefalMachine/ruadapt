"""Gradient checkpointing vs batch-size/chunk-len balance for Qwen3.5-27B SFT.

Two regimes:
  * packed:  B=1, documents packed with position_ids reset + seq_idx + cu_seqlens.
             FLA chunk_gated_delta_rule REQUIRES batch_size=1 when cu_seqlens is
             given, so in packed mode the effective batch size is the chunk length S.
  * dense:   B>=2 with attention_mask (current production shape, dynamic padding).

Sweeps {grad ckpt ON/OFF} x {shape}. Optimizer states are materialized so peak
memory is realistic.

Run manually on a free GPU:
    CUDA_VISIBLE_DEVICES=4 python tests/perf/bench_ckpt_balance.py
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
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_3",
        trust_remote_code=True,
    )
    print(f"loaded in {time.perf_counter()-t0:.1f}s", flush=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(
        model, LoraConfig(r=128, lora_alpha=128, lora_dropout=0.0, target_modules=LORA_TARGETS)
    )
    model.print_trainable_parameters()
    from transformers.integrations.liger import apply_liger_kernel
    apply_liger_kernel(model, None)
    print("liger fwd:", model.get_base_model().forward.__qualname__, flush=True)
    model = model.to("cuda")
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4, fused=True
    )
    print(f"on device, allocated: {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    return model, optimizer


def mk_packed(S, docs=2):
    ids = torch.randint(0, VOCAB - 1000, (1, S), device="cuda")
    length = S // docs
    pos, sid, cu = [], [], [0]
    acc, doc = 0, 0
    for _ in range(docs):
        pos.append(torch.arange(length))
        sid.extend([doc] * length)
        acc += length
        cu.append(acc)
        doc += 1
    return dict(
        input_ids=ids,
        labels=ids.clone(),
        position_ids=torch.cat(pos).unsqueeze(0).contiguous().cuda(),
        seq_idx=torch.tensor([sid], device="cuda", dtype=torch.int32),
        cu_seq_lens_q=torch.tensor(cu, device="cuda", dtype=torch.int32),
    )


def mk_dense(B, S):
    ids = torch.randint(0, VOCAB - 1000, (B, S), device="cuda")
    return dict(
        input_ids=ids,
        labels=ids.clone(),
        attention_mask=torch.ones(B, S, dtype=torch.long, device="cuda"),
        position_ids=torch.arange(S, device="cuda").unsqueeze(0).expand(B, -1).contiguous(),
    )


def bench(model, optimizer, name, ckpt, inputs, ntok, iters=5, warmup=2):
    if ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_disable()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        for i in range(warmup):
            out = model(**inputs)
            out.loss.backward()
            if i == 0:
                optimizer.step()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            out = model(**inputs)
            out.loss.backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        toks = ntok / dt
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
    del inputs


def main():
    model, optimizer = build_model()
    # Packed (B=1 required by FLA varlen): chunk length = effective batch size
    bench(model, optimizer, "P ckptON  S2048", True, mk_packed(2048), 2048)
    bench(model, optimizer, "P ckptOFF S2048", False, mk_packed(2048), 2048)
    bench(model, optimizer, "P ckptON  S4096", True, mk_packed(4096), 4096)
    bench(model, optimizer, "P ckptOFF S4096", False, mk_packed(4096), 4096)
    bench(model, optimizer, "P ckptON  S8192", True, mk_packed(8192), 8192)
    bench(model, optimizer, "P ckptOFF S8192", False, mk_packed(8192), 8192)
    # Dense (current production style), B=2
    bench(model, optimizer, "D ckptON  B2 S2048", True, mk_dense(2, 2048), 2 * 2048)
    bench(model, optimizer, "D ckptOFF B2 S2048", False, mk_dense(2, 2048), 2 * 2048)
    print("done", flush=True)


if __name__ == "__main__":
    main()
