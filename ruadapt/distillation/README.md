# SFT Distillation: Scripts Overview

This repo implements **knowledge distillation for the SFT (supervised fine-tuning) step** using *offline* teacher logits. It’s a two-stage pipeline:

1. **Extract Top-K teacher logits** and save them to disk.  
2. **Distill a student** on SFT data using teacher's saved logits.

---

## Structure

### 1) `distillation/logits_ddp.py` — Teacher Logits Extractor
Generates and saves **Top-K logits** from a teacher model. Designed for multi-GPU with PyTorch Distributed.

**Key features**
- **Input**: reads JSON/JSONL via `datasets.load_dataset('json', ...)`.
- **Chat-format aware**: builds inputs with `tokenizer.apply_chat_template(...)`.
- **Filtering**:
  - Skips malformed / multi-assistant examples.
  - Drops samples exceeding `--max_length`.
  - Optionally sorts by input length (`--sort_by_len`).
  - Writes the kept, **order-aligned** records to `--filtered_data_file` (JSONL).

- **Output (Parquet)**:
  - Schema:
    - `idx: int64` – original sample index
    - `topk_indices: list<int32>` 
    - `topk_values: list<float32>` 

**CLI (example)**
```bash
torchrun --nnodes=2 \
         --nproc-per-node=8 \
         -m distillation.logits_ddp \
         --model_name 'RefalMachine/RuadaptQwen3-32B-Instruct' \
         --input_file data/sft_train_data.jsonl \
         --output_file data/teacher_train_top100.parquet \
         --filtered_data data/filtered_sft_train.jsonl \
         --max_length 8192 \
         --batch_size 1 \
         --sort_by_len \
         --topk 100
```

### 2) `distillation/distill_sft_new.py` — Student Training with Offline Distillation
This script fine-tunes a **student model** using both supervised SFT loss and **knowledge distillation (KD)** from pre-computed teacher logits. It uses the filtered SFT dataset and the teacher’s Top-K logits produced by `logits_ddp.py`.

**Key features**
- **Configuration-driven**: training parameters, model settings, and dataset paths are defined in a JSON config (`--config_file`).
- **Training objective**:
  - **Supervised loss**: next-token prediction via cross-entropy on assistant tokens.
  - **Distillation loss**:
    - KL divergence between teacher softmax distribution and student log-softmax assigned with Top-K indices.
    - Optionally applies adaptive weighting based on teacher–student agreement on the hard target.
  - Final objective:  
    `loss = SFT_loss + loss_multi * KD_loss`
- **Outputs**:
  - Saves the fine-tuned student model and tokenizer to `--output_dir`.

**CLI (example)**
```bash
torchrun --nnodes=1 \
         --nproc-per-node=8 \
         distill_sft_new.py \
         --config_file configs/distill_config.json
         --output_dir models/Qwen3-4B_distilled
```


### Data & Formats

#### SFT data (input)
- JSONL where each line has a messages array of chat turns:
```bash 
{"messages":[{"role":"user","content":"..."},
             {"role":"assistant","content":"..."},
             {"role":"user","content":"..."}]}
```

- Roles like "bot" are normalized to "assistant".
- The extractor will write a filtered JSONL (`--filtered_data_file`) that stays index-aligned with the Parquet logits.

#### Teacher logits (output of stage 1, input to stage 2)
-	Parquet with columns:
	-	idx (`int64`)
	-	topk_indices (`list<int32>`)
	-	topk_values  (`list<float32>`)



#### Minimal Config Example (configs/distill_config.json)
```bash
{
    "trainer": {
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "eval_steps": 16,   
        "save_steps": 800,
        "logging_steps": 1,
        "learning_rate": 1e-04,
        "num_train_epochs": 2,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 16,
        "bf16": true,
        "fp16": false,
        "optim": "adamw_torch_fused",
        "load_best_model_at_end": false,
        "save_total_limit": 1,
        "seed": 1337,
        "max_grad_norm": 1.0,
        "weight_decay": 0.05,
        "ddp_find_unused_parameters": false,
        "eval_strategy": "steps"
    },
    "lora": {
        "r": 128,
        "lora_alpha": 128,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        "modules_to_save": [
            "lm_head"
        ]
    },
    "train": {
        "data_path":      "data/train_maxlen8192_top10.jsonl",
        "teacher_logits": "distillation/RuadaptQwen3-32B-Instruct_logits/teacher_train_maxlen8192_top10.parquet"
    },
    "validation": {
        "data_path":      "data/test_maxlen8192_top10.jsonl",
        "teacher_logits": "distillation/RuadaptQwen3-32B-Instruct_logits/teacher_test_maxlen8192_top10.parquet"
    },
    "loss_multi": 0.1,
    "load_in_8bit": false,
    "gradient_checkpointing": true,
    "load_in_4bit": false,
    "only_target_loss": true,
    "max_tokens_count": 8192,
    "max_seq_length": 8192,
    "bos_token": "<|endoftext|>",
    "eos_token": "<|im_end|>",
    "pad_token": "<|endoftext|>",
    "model_name": "/qwen3/4b/ruadapt_qwen3_4B_ext_u48_290425_part1_lr5e4_wsd_bs256_part2_5e5_128_512_bs128_straight_as2.0_instruct/lep",
    "per_token": true
}

```

### Typical Workflow
**1.	Extract teacher logits**
- Run logits_ddp.py on your SFT data to create teacher_logits.parquet and filtered_data.jsonl.

**2.	Distill the student**
- Use distill_sft_new.py with created files (pass them via config) and train model.

