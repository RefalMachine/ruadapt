# Deprecated Code

This directory contains code from the old architecture that has not yet been migrated to the new `ruadapt` package structure.

**Do not import from here in new code.** This directory is for reference only.

---

## `instruct_tuning/`

Old instruction tuning code. Uses **Unsloth** for training, replaced by `ruadapt.training` (from dgx_llm).

| Script | Method | Status |
|--------|--------|--------|
| `train_sft.py` | SFT (Unsloth) | Replaced by training core + SFT factory |
| `train_sft_transformers.py` | SFT (Transformers + PEFT) | Reference for SFT factory improvements |
| `train_sft_fsdp.py` | SFT (FSDP, 32B+) | Reference for FSDP patterns |
| `train_dpo.py` | DPO (Unsloth + trl) | Future: extend training core |
| `train_kto.py` | KTO (Unsloth + trl) | Future: extend training core |
| `train_cpo.py` | CPO (Unsloth + trl) | Future: extend training core |
| `train_smpo.py` | SimpleMarginPO (Unsloth) | Future: extend training core |
| `smpo_trainer.py` | Custom SMPO trainer (1047 LOC) | **Valuable** — port to `ruadapt/training/trainers/smpo.py` when needed |
| `dataset.py` | ChatDataset | Reference for SFT factory |
| `dpo_dataset.py` | DPODataset | Reference for future DPO factory |
| `utils.py` | `set_random_seed()` | Migrated to `ruadapt/utils/seed.py` |
| `models_configs/` | Per-model training configs | Reference only |
| `datasets/` | Dataset composition scripts | Reference only |

## `pretraining/`

Old HF Trainer-based pretraining. Replaced by `ruadapt.training.train`.

| Script | Status |
|--------|--------|
| `train_trainer.py` | Replaced by training core |
| `utils.py` | Some utils reusable (`custom_tokenize`, `group_texts`) |
| `run_train_ruadapt.sh` | Old launch script |

## `root_scripts/`

Old pipeline orchestrators and shell scripts. These reference old module paths and Unsloth-based code.

| Script | Purpose | Status |
|--------|---------|--------|
| `run_pipeline.py` | Main pipeline orchestrator | Replaced by shell scripts in `scripts/` |
| `run_pipeline_config.py` | Config-driven pipeline | Replaced by `scripts/` |
| `run_pipeline_app.py` | Gradio app for pipeline | Deprecated |
| `run_pipeline_infer.py` | Inference pipeline | Replaced by `ruadapt/inference/vllm.py` |
| `extend_or_replace.py` | Tokenizer extend/replace | Replaced by `ruadapt/tokenization/` |
| `mix_data.py` | Data mixing utility | Deprecated |
| `process_checkpoints.sh` | Checkpoint processing | Deprecated |
| `process_loras.sh` | LoRA processing | Deprecated |
| `run_extend_or_replace.sh` | Tokenizer extension | Deprecated |
| `run_extend_or_replace_unigram.sh` | Unigram extension | Deprecated |
| `run_extension_pipeline.sh` | Full extension pipeline | Deprecated |
| `run_extension_pipeline_example.sh` | Example pipeline | Deprecated |
| `run_pipeline.sh` | Pipeline launcher | Deprecated |

## `pipeline_configs/`

Old pipeline step configuration JSONs (19 files). These reference old module paths and are kept for reference only.

## Migration Status

See [../MIGRATION_AND_REFACTORING_PLAN.md](../MIGRATION_AND_REFACTORING_PLAN.md) for the full migration plan.
