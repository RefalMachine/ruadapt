import argparse
import json
import math
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForCausalLM,
    Trainer, TrainingArguments
)
from tqdm import tqdm


# ════════════════════════════════════════════════
# Dataset
# ════════════════════════════════════════════════

class PackedDatasetWithLossMask(Dataset):
    def __init__(self, texts, tokenizer, max_length=512, text_truncation_length=None):
        print(f"Tokenizing {len(texts)} documents...")
        encodings = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
        
        eos_id = tokenizer.eos_token_id
        bos_id = getattr(tokenizer, 'bos_token_id', None)
        
        all_tokens = []
        # loss_mask: 1 = считать loss, 0 = игнорировать (label=-100)
        all_mask = []
        
        n_truncated = 0
        n_complete = 0
        
        for ids in tqdm(encodings["input_ids"], desc="Packing"):
            if len(ids) == 0:
                continue
            
            is_truncated = (
                text_truncation_length is not None
                and len(ids) >= text_truncation_length
            )
            
            if bos_id is not None:
                all_tokens.append(bos_id)
                # BOS после обрезанного текста — не считаем loss
                # (модель не могла предсказать BOS из контекста обрыва)
                # BOS после полного текста (после EOS) — тоже спорно,
                # но менее вредно. Маскируем все BOS для простоты.
                all_mask.append(0)
            
            # Токены документа
            all_tokens.extend(ids)
            all_mask.extend([1] * len(ids))
            
            if is_truncated:
                # Последний токен обрезанного текста: 
                # его label (= следующий токен) — это BOS другого документа.
                # Это неверный сигнал → маскируем
                all_mask[-1] = 0
                n_truncated += 1
            else:
                all_tokens.append(eos_id)
                all_mask.append(1)  # EOS после полного текста — валидный label
                n_complete += 1
        
        print(f"Documents: {n_complete} complete, {n_truncated} truncated")
        
        total = len(all_tokens)
        n_chunks = total // max_length
        usable = n_chunks * max_length
        
        self.data = torch.tensor(
            all_tokens[:usable], dtype=torch.long
        ).reshape(n_chunks, max_length)
        
        self.mask = torch.tensor(
            all_mask[:usable], dtype=torch.long
        ).reshape(n_chunks, max_length)
        
        masked_count = (self.mask == 0).sum().item()
        self.total_tokens = usable
        print(f"Packed: {n_chunks} chunks × {max_length} = {usable:,} tokens")
        print(f"Masked (no loss): {masked_count:,} tokens ({100*masked_count/usable:.1f}%)")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        return {
            "input_ids": self.data[i],
            "loss_mask": self.mask[i],
        }


class ClmCollatorWithMask:
    """
    Создаёт labels из input_ids, маскирует позиции где loss_mask=0 
    через label=-100.
    """
    def __call__(self, features):
        input_ids = torch.stack([f["input_ids"] for f in features])
        loss_mask = torch.stack([f["loss_mask"] for f in features])
        
        # Labels для CLM: сдвигаем на 1 (input[t] → предсказываем input[t+1])
        # HF модели делают сдвиг внутри forward(), поэтому labels = input_ids
        # Но маску нужно тоже сдвинуть:
        # label[t] = input_ids[t], mask для label[t] определяет,
        # считаем ли loss на предсказание токена t.
        #
        # Если input[t-1] — последний токен обрезанного документа,
        # то label[t] (= input[t]) — это BOS следующего документа.
        # Мы хотим замаскировать label[t], а mask[t] уже = 0 (мы маскировали BOS).
        # 
        # Если input[t] — последний токен обрезанного документа (mask[t]=0),
        # то label[t+1] (= input[t+1]) — BOS, mask[t+1] = 0. Тоже ок.
        #
        # Всё корректно: маска на input позициях соответствует маске на labels.
        
        labels = input_ids.clone()
        labels[loss_mask == 0] = -100
        
        return {"input_ids": input_ids, "labels": labels}


# ════════════════════════════════════════════════
# Anti-forgetting hooks
# ════════════════════════════════════════════════

def register_freeze_hooks(model, freeze_idx):
    """
    Обнуляет градиенты для токенов с id < freeze_idx
    в embed И lm_head (если не tied).
    """
    def mask_grad_hook(grad):
        limit = min(freeze_idx, grad.shape[0])
        with torch.no_grad():
            grad[:limit, :] = 0.0
        return grad
    
    embeds = model.get_input_embeddings()
    embeds.weight.register_hook(mask_grad_hook)
    print(f"[Hook] embed_tokens: freezing rows 0..{freeze_idx-1}")
    
    lm_head = model.get_output_embeddings()
    if lm_head is not None and lm_head.weight is not embeds.weight:
        lm_head.weight.register_hook(mask_grad_hook)
        print(f"[Hook] lm_head: freezing rows 0..{freeze_idx-1}")
    else:
        print(f"[Hook] lm_head: tied with embed_tokens, single hook sufficient")


# ════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--eval_data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="./cpt_results")
    
    # Training budget
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--epochs', type=float, default=1.0)
    parser.add_argument('--seq_length', type=int, default=512)
    
    # Batch size
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Per-device batch size")
    parser.add_argument('--grad_accum', type=int, default=8)
    
    # Optimizer
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help="WD=0 recommended for embed-only CPT")
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.95)
    parser.add_argument('--adam_epsilon', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    
    # Scheduler
    parser.add_argument('--scheduler', type=str, default="cosine",
                        choices=["cosine", "constant_with_warmup"])
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--min_lr_ratio', type=float, default=0.01,
                        help="Minimum LR as fraction of peak (for cosine)")
    
    # Eval & logging
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_steps', type=int, default=5000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    
    # Freeze
    parser.add_argument('--freeze_idx', type=int, default=248044,
                        help="Token IDs below this are frozen")
    
    args = parser.parse_args()

    # ── Model ──────────────────────────────────
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Detect model class
    architectures = getattr(config, "architectures", [])
    if architectures and "Qwen3_5ForConditionalGeneration" in architectures:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
        ModelClass = Qwen3_5ForConditionalGeneration
    else:
        ModelClass = AutoModelForCausalLM
    ModelClass = AutoModelForCausalLM
    model = ModelClass.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    # ── Freeze everything, unfreeze embed + head ──
    for param in model.parameters():
        param.requires_grad = False
    
    embeds = model.get_input_embeddings()
    embeds.weight.requires_grad = True
    
    lm_head = model.get_output_embeddings()
    if lm_head is not None and lm_head.weight is not embeds.weight:
        lm_head.weight.requires_grad = True

    register_freeze_hooks(model, args.freeze_idx)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} "
          f"({100*trainable/total:.2f}%)")

    # ── Data ───────────────────────────────────
    print(f"Loading data...")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        train_texts = [
            item['text'] if isinstance(item, dict) else item 
            for item in json.load(f)
        ]
    with open(args.eval_data_path, 'r', encoding='utf-8') as f:
        eval_texts = [
            item['text'] if isinstance(item, dict) else item 
            for item in json.load(f)
        ]

    train_dataset = PackedDatasetWithLossMask(train_texts, tokenizer, args.seq_length, text_truncation_length=512)
    eval_dataset = PackedDatasetWithLossMask(eval_texts, tokenizer, args.seq_length, text_truncation_length=512)
    collator = ClmCollatorWithMask()

    # ── Compute budget info ────────────────────
    n_gpus = max(1, torch.cuda.device_count())
    effective_bs = args.batch_size * args.grad_accum * n_gpus
    tokens_per_step = effective_bs * args.seq_length
    
    if args.steps is not None:
        total_steps = args.steps
        total_tokens = total_steps * tokens_per_step
    else:
        steps_per_epoch = len(train_dataset) // effective_bs
        total_steps = int(steps_per_epoch * args.epochs)
        total_tokens = total_steps * tokens_per_step
    
    print(f"\n{'='*50}")
    print(f"Training config:")
    print(f"  GPUs:             {n_gpus}")
    print(f"  Per-device BS:    {args.batch_size}")
    print(f"  Grad accum:       {args.grad_accum}")
    print(f"  Effective BS:     {effective_bs} sequences")
    print(f"  Seq length:       {args.seq_length}")
    print(f"  Tokens/step:      {tokens_per_step:,}")
    print(f"  Total steps:      {total_steps:,}")
    print(f"  Total tokens:     {total_tokens:,}")
    print(f"  Warmup steps:     {int(total_steps * args.warmup_ratio):,}")
    print(f"  LR:               {args.lr}")
    print(f"  Scheduler:        {args.scheduler}")
    print(f"  Adam betas:       ({args.adam_beta1}, {args.adam_beta2})")
    print(f"  Adam eps:         {args.adam_epsilon}")
    print(f"  Weight decay:     {args.weight_decay}")
    print(f"{'='*50}\n")

    # ── Training ───────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        
        # Budget
        max_steps=args.steps if args.steps is not None else -1,
        num_train_epochs=args.epochs if args.steps is None else 1,
        
        # Batch
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,  # eval не нужен grad → можно больше
        gradient_accumulation_steps=args.grad_accum,
        
        # Optimizer
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        
        # Scheduler
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup_ratio,
        
        # Precision
        bf16=True,
        
        # Logging & eval
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        
        # Saving
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,
        
        # Misc
        report_to="tensorboard",
        dataloader_num_workers=args.workers,
        dataloader_pin_memory=True,
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        
        # Не нужно для frozen model
        gradient_checkpointing=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    print("Starting CPT...")
    trainer.train()

    # ── Save ───────────────────────────────────
    is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    if is_main:
        save_path = args.output_dir
        print(f"Saving to {save_path}...")
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)
        
        # Сохраняем конфиг эксперимента
        with open(f"{save_path}/cpt_config.json", 'w') as f:
            json.dump(vars(args), f, indent=2)

    print("CPT completed.")


if __name__ == "__main__":
    main()