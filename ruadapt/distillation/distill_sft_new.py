import os
import argparse
import torch
import json
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    BitsAndBytesConfig,
    default_data_collator,
    DataCollatorWithPadding,
)
import numpy as np
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
import pandas as pd
import torch.distributed as dist
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

def print_token_rows(inp_ids, labels, attn, topk_idx, topk_val, tokenizer, start=0, end=20):
    L, K = topk_idx.shape
    end = min(end, L)
    probs = torch.softmax(topk_val, dim=-1)
    print(f"{'pos':>4} {'A':>1} {'inp_id':>7} {'tok':<15} {'lab':>7} {'next_t':>9} {'top1':>7} {'p1':>6} {'mn':>2} {'ms':>2}")
    print("-"*90)
    for t in range(start, end):
        a = int(attn[t])
        inp_id = int(inp_ids[t])
        tok = tokenizer.decode([inp_id])[:15]
        lab = int(labels[t]) if labels is not None else -100
        next_t = int(inp_ids[t+1]) if t+1 < L else -1
        top1 = int(topk_idx[t,0])
        p1 = float(probs[t,0])
        mn = 'y' if top1 == next_t else 'n'
        ms = 'y' if top1 == inp_id else 'n'
        print(f"{t:4d} {a:1d} {inp_id:7d} {tok:<15} {lab:7d} {next_t:9d} {top1:7d} {p1:6.3f} {mn:>2} {ms:>2}")
    print("-"*90)

    show_t = start
    ids_row  = topk_idx[show_t].tolist()
    probs_row= probs[show_t].tolist()
    decoded  = [tokenizer.decode([tid]) for tid in ids_row]
    print(f"Top-K ids at t={show_t}: {ids_row}")
    print(f"Top-K probs at t={show_t}: {[round(p,4) for p in probs_row]}")
    print(f"Decoded tokens: {decoded}")


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

class DistillationDataset(Dataset):
    def __init__(self, data_path: str, teacher_logits_path: str, tokenizer, max_length: int = 512, 
                only_target_loss: bool = True,
                add_global_bos: bool = False,
                add_global_eos: bool = False,
                labels_pad_token_id: int = -100,
                ):
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.only_target_loss = only_target_loss
        self.labels_pad_token_id = labels_pad_token_id
        self.add_global_bos = add_global_bos
        self.add_global_eos = add_global_eos

        with open(data_path, 'r', encoding='utf-8') as f:
            self.data_records = [json.loads(line) for line in f]
            if is_main_process():
                print("Original data read completed successfully!")

        #TODO Remove and replace with load_dataset
        if teacher_logits_path.lower().endswith('.parquet'):
            
            self.teacher_records = load_dataset("parquet", data_files=teacher_logits_path, split='train')

            features = self.teacher_records.features
            if 'topk_indices' not in features or 'topk_values' not in features:
                raise ValueError("Parquet must contain 'topk_indices' and 'topk_values' columns")

            if is_main_process():
                print(f"Teacher's logits opened via datasets (parquet): {teacher_logits_path}")

            #df = pd.read_parquet(teacher_logits_path)
            #self.teacher_records = []
            #for rec in df.to_dict(orient='records'):
            #    self.teacher_records.append(rec)

            #for rec in self.teacher_records:
            #    raw_idx = rec['topk_indices']
            #    idx_list = list(raw_idx)
            #    idx_arrays = [np.asarray(x, dtype=np.int64) for x in idx_list]
            #    rec['topk_indices'] = np.stack(idx_arrays, axis=0)


            #    raw_val = rec['topk_values']
            #    val_list = list(raw_val)
            #    val_arrays = [np.asarray(x, dtype=np.float32) for x in val_list]
            #    rec['topk_values'] = np.stack(val_arrays, axis=0)
          
            #if is_main_process():
            #    print(f"Teacher's logits read from parquet {teacher_logits_path} successfully!")
        else:
            with open(teacher_logits_path, 'r', encoding='utf-8') as f:
                self.teacher_records = [json.loads(line) for line in f]
                if is_main_process():
                    print("Teacher's logits read from jsonl successfully!")

        assert len(self.data_records) == len(self.teacher_records), \
            f"Data and teacher logits length mismatch: {len(self.data_records)} vs {len(self.teacher_records)}"

        #bad_idxs = [
        #    i for i, t in enumerate(self.teacher_records)
        #    if 'topk_indices' not in t or 'topk_values' not in t
        #]
        #if bad_idxs:
        #    raise ValueError(f"Teacher logits missing at indices: {bad_idxs}")
        
        #if is_main_process():
        #    print(f"All {len(self.teacher_records)} teacher records have required keys")

        self.valid_indices = []
        for i, rec in enumerate(self.data_records):
            if self.convert_record(rec) is not None:
                self.valid_indices.append(i)
        
        if is_main_process():
            print(f"Kept {len(self.valid_indices)} / {len(self.data_records)} usable samples (others had no assistant tokens or were empty).")

    def get_tokens(self, messages):
        tokens = self.tokenizer.apply_chat_template(
            messages,
            add_special_tokens=False,
            tokenize=True,
            add_generation_prompt=False,
        )
        if tokens and tokens[0] == self.tokenizer.bos_token_id:
            tokens = tokens[1:]
        return tokens

    def convert_record(self, record):
        input_ids_list, labels_list = [], []
        for message in record['messages']:
            if message.get('role') == 'bot':
                message['role'] = 'assistant'

            message_input_ids = self.get_tokens([message])
            message_labels = list(message_input_ids)
            if len(input_ids_list) + len(message_input_ids) > self.max_length - 2:
                break

            labels_mask = [self.labels_pad_token_id] * len(message_input_ids)
            if message['role'] not in ('assistant', 'bot', 'gpt') and self.only_target_loss:
                message_labels = labels_mask

            input_ids_list.extend(message_input_ids)
            labels_list.extend(message_labels)

        if not input_ids_list:
            return None
        
        if self.add_global_bos and input_ids_list[0] != self.tokenizer.bos_token_id:
            input_ids_list.insert(0, self.tokenizer.bos_token_id)
            labels_list.insert(0, self.labels_pad_token_id)

        if self.add_global_eos and input_ids_list[-1] != self.tokenizer.eos_token_id:
            input_ids_list.append(self.tokenizer.eos_token_id)
            labels_list.append(self.tokenizer.eos_token_id)

        if all(l == self.labels_pad_token_id for l in labels_list):
            return None
        
        input_ids = torch.LongTensor(input_ids_list)
        labels = torch.LongTensor(labels_list)
        attention_mask = input_ids.new_ones(input_ids.size())
        return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        record = self.data_records[real_idx]
        tokens = self.convert_record(record)

        if tokens is None:
            return self.__getitem__((idx + 1) % len(self.valid_indices))

        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        labels = tokens['labels']
        
        if self.teacher_records is not None:
            teacher = self.teacher_records[real_idx]
            idx_list = teacher['topk_indices']
            val_list = teacher['topk_values']
            if len(idx_list) == 0:
                raise ValueError("topk fields shouldn't be empty")
            else:
                topk_indices = torch.tensor(idx_list, dtype=torch.long)
                topk_values = torch.tensor(val_list, dtype=torch.float32)
        
        #else:
        #    teacher = self.teacher_records[real_idx]
        #
        #    teacher_indices = teacher['topk_indices']
        #    teacher_values = teacher['topk_values']
        #    if isinstance(teacher_indices, list) and len(teacher_indices) == 0:
        #        raise ValueError("topk fields shouldn't be empty")
        #    else:
        #        topk_indices = torch.as_tensor(teacher_indices, dtype=torch.long)
        #        topk_values = torch.as_tensor(teacher_values, dtype=torch.float32)

        item = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'topk_indices': topk_indices,
            'topk_values':  topk_values,
            #'teacher_mask': torch.tensor(teacher['mask'], dtype=torch.bool),
            #'next_ids': torch.tensor(teacher['next_ids'], dtype=torch.long),
            '_idx': real_idx
        }

        #k_idx = item['topk_indices']
        #k_val = item['topk_values']
        #k_idx = torch.cat([torch.zeros((1, k_idx.size(1)), dtype=k_idx.dtype), k_idx[:-1]], dim=0)
        #k_val = torch.cat([torch.zeros((1, k_val.size(1)), dtype=k_val.dtype), k_val[:-1]], dim=0)
        #item['topk_indices'] = k_idx
        #item['topk_values'] = k_val
        return item



class DistillTrainer(Trainer):
    def __init__(self, per_token: bool, loss_multi: float = 0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.per_token = per_token
        self.loss_multi = loss_multi
        self._debug_logged_steps = 0
        self._debug_log_path = os.path.join(os.getcwd(), "debug_loss_breakdown-labels_shift.txt")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            
        outputs = model(input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'])
        student_logits = outputs.logits  # [B, L, V]
        
        #if self._debug_logged_steps < 10:
        #    if is_main_process():
        #        with open(self._debug_log_path, "a") as f:
        #            f.write(f"=== DEBUG STEP {self._debug_logged_steps} - full inputs ===\n")
        #            f.write(str(inputs))
        #            f.write("\n")

        teacher_idx  = inputs['topk_indices']
        teacher_vals = inputs['topk_values']
        #loss_mask    = inputs['teacher_mask'][:, 1:]           
        # (next_ids = inputs['next_ids'][:, 1:])

        if 'labels' in inputs:
            assistant_mask = (inputs['labels'] != -100)  
        else:
            assistant_mask = inputs['attention_mask'].bool()

        #pad_mask = inputs['teacher_mask'][:, 1:]
        loss_mask = assistant_mask 

        if self.per_token:
            # New test approach

            B, L, V = student_logits.shape
            labels_in = inputs['labels']
            next_labels = torch.cat([
                labels_in[:, 1:],
                torch.full((B, 1), -100, device=labels_in.device, dtype=torch.long)
            ], dim=1)


            student_topk   = student_logits.gather(dim=-1, index=teacher_idx)
            teacher_probs = F.softmax(teacher_vals, dim=-1)
            student_logprobs = F.log_softmax(student_topk, dim=-1)

            kl_full = F.kl_div(student_logprobs, teacher_probs, reduction="none")  
            kl_token = kl_full.sum(dim=-1)  

            hard = next_labels
            hard_safe = torch.where(hard == -100, torch.zeros_like(hard), hard)
            s_hard_raw = student_logits.gather(-1, hard_safe.unsqueeze(-1)).squeeze(-1)  
            s_hard = torch.where(hard == -100, torch.zeros_like(s_hard_raw), s_hard_raw)

            match = (teacher_idx == hard_safe.unsqueeze(-1))                      
            t_hard_raw = (teacher_vals * match.float()).sum(dim=-1)               
            in_topk = match.any(dim=-1) & (hard != -100)
            t_hard = torch.where(in_topk, t_hard_raw, torch.zeros_like(t_hard_raw))

            ratio = (s_hard.detach() / (t_hard.detach().abs() + 1e-5))
            w = (1.0 - torch.exp(-ratio)).clamp(0, 1)                             

            kd_mask = assistant_mask
            kd_loss = (kl_token * w * kd_mask).sum() / kd_mask.sum().clamp_min(1)

            ce_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
            sft_loss = ce_fct(
                student_logits.contiguous().view(-1, V),
                next_labels.contiguous().view(-1)
            )

            loss = sft_loss + self.loss_multi * kd_loss


            # Old working approach
            """
            student_topk   = student_logits.gather(dim=-1, index=teacher_idx)
            teacher_logits = teacher_vals                            

            teacher_probs = F.softmax(teacher_logits, dim=-1)
            student_logprobs = F.log_softmax(student_topk, dim=-1)

            kl_full = F.kl_div(student_logprobs, teacher_probs, reduction="none")  

            kl_token = kl_full.sum(dim=-1)

            kl_token = kl_token * loss_mask
            kd_loss = kl_token.sum() / loss_mask.sum().clamp_min(1)

            B, L, V = student_logits.shape

            input_ids = inputs['labels']
            next_labels = torch.cat([
                input_ids[:, 1:],
                torch.full((B, 1), -100, device=input_ids.device, dtype=torch.long)
            ], dim=1)

            ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')

            sft_loss = ce_loss(
                student_logits.contiguous().view(-1, V),
                next_labels.contiguous().view(-1)
            )

            alpha = 0.1
            loss = kd_loss + alpha * sft_loss
            """
            
            if self._debug_logged_steps < 10:

                kl_token_raw = kl_full.sum(dim=-1)              
                masked_kl_token = kl_token_raw * loss_mask        
                sum_before_mask = kl_token_raw.sum(dim=-1)        
                sum_after_mask = masked_kl_token.sum(dim=-1)      
                denom = loss_mask.sum().clamp_min(1)        
                averaged_loss = sum_after_mask / denom         

                B, L = kl_token_raw.shape
                dump = []

                student_topk_vals, student_topk_ids = student_logits.topk(10, dim=-1)  

                def maybe_index(tensor, b):
                    return tensor.item() if tensor.dim() == 0 else tensor[b].item()

                for b in range(B):
                    example = {
                        "example_index": b,
                        "sum_before_mask": float(maybe_index(sum_before_mask, b)),
                        "sum_after_mask": float(maybe_index(sum_after_mask, b)),
                        "num_masked_positions": int(maybe_index(denom, b)),
                        "averaged_loss": float(maybe_index(averaged_loss, b)),
                        "positions": []                
                    }
                    for t in range(L):
                        pos_entry = {
                            "t": t,
                            "input_id": int(inputs["input_ids"][b, t].item()),
                            "label_id": int(next_labels[b, t].item()),
                            "attention": int(inputs["attention_mask"][b, t].item()),
                            "loss_mask": int(loss_mask[b, t].item()),
                            "kl_token_raw": float(kl_token_raw[b, t].item()),
                            "masked_kl_token": float(masked_kl_token[b, t].item()),
                            "teacher_topk_ids": inputs["topk_indices"][b, t].tolist(),
                            "teacher_topk_vals": [float(v) for v in inputs["topk_values"][b, t].tolist()],
                            "student_topk_vals": [float(v) for v in student_topk[b, t].tolist()],
                            "student_topk_ids": student_topk_ids[b, t].tolist()
                        }
                        example["positions"].append(pos_entry)
                    dump.append(example)

                if is_main_process():
                    with open(self._debug_log_path, "a") as f:
                        f.write(f"=== DEBUG STEP {self._debug_logged_steps} - per_token breakdown ===\n")
                        f.write(json.dumps(dump, ensure_ascii=False, indent=2))
                        f.write(f"\nComputed kd_loss scalar: {float(kd_loss.item())}\n\n")
                    self._debug_logged_steps += 1
            
        else:
            last_logits = outputs.logits[:, -1, :]                         
            student_topk = last_logits.gather(dim=-1, index=inputs['topk_indices'][:, -1, :])  
            teacher_logits = inputs['topk_values'][:, -1, :]                                   

            teacher_probs = F.softmax(teacher_logits, dim=-1).float()
            student_logprob = F.log_softmax(student_topk, dim=-1).float()

            loss = F.kl_div(student_logprob, teacher_probs, reduction="batchmean")

        #if self._debug_logged_steps > 10:
        #    raise(123)

        return (loss, outputs) if return_outputs else loss
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file",    type=str)
    parser.add_argument("--output_dir",     type=str)
    parser.add_argument("--num_workers",    type=int, default=4)
    args = parser.parse_args()

    with open(args.config_file, 'r') as cf:
        config = json.load(cf)

    trainer_cfg = config['trainer']
    trainer_cfg['output_dir'] = args.output_dir
    hf_parser = HfArgumentParser((TrainingArguments,))
    training_args = hf_parser.parse_dict(trainer_cfg)[0]

    training_args.dataloader_num_workers = args.num_workers
    training_args.dataloader_pin_memory = True
    training_args.remove_unused_columns = False

    model_name = config['model_name']
    
    bnb_config = None
    if config['load_in_4bit']:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    elif config['load_in_8bit']:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    student = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=f"cuda:{local_rank}",
    )

    if config.get('gradient_checkpointing', False):
        student.gradient_checkpointing_enable()
        student.config.use_cache = False

    if 'lora' in config:
        lora_config = LoraConfig(**config['lora'])
        if student.config.tie_word_embeddings and 'lm_head' in config['lora']['modules_to_save']:
            if 'embed_tokens' in config['lora']['modules_to_save']:
                lora_config.modules_to_save = ['lm_head']
                config['lora']['modules_to_save'] = ['lm_head']

        student = get_peft_model(student, lora_config)
        if student.config.tie_word_embeddings and 'lm_head' in config['lora']['modules_to_save']:
            student.base_model.model.model.embed_tokens.weight = student.base_model.model.lm_head.modules_to_save["default"].weight
    else:
        for param_name, param in student.model.named_parameters():
            if 'embed_tokens' not in param_name:
                param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = 'left'
    tokenizer.add_special_tokens({
        'bos_token': config['bos_token'],
        'eos_token': config['eos_token'],
        'pad_token': config['pad_token']
    })

    def collate_fn(features):
        max_len = max(f["input_ids"].size(0) if isinstance(f["input_ids"], torch.Tensor) else len(f["input_ids"]) for f in features)
        K = features[0]["topk_indices"].size(1)

        def pad_and_trunc(x, length, pad_value, dim=0):
            curr = x.size(dim) if torch.is_tensor(x) else len(x)
            pad_len = length - curr
            #print(pad_len)
            if pad_len < 0:
                if tokenizer.padding_side == "left":
                    if torch.is_tensor(x):
                        return x[-length:] if x.dim() == 1 else x[-length:, ...]
                    else:
                        return x[-length:]
                else:
                    if torch.is_tensor(x):
                        return x[:length] if x.dim() == 1 else x[:length, ...]
                    else:
                        return x[:length]
            if pad_len > 0:
                if torch.is_tensor(x):
                    if dim != 0:
                        raise ValueError("pad_and_trunc: can only pad along dim=0 for tensors")
                    if tokenizer.padding_side == 'left':
                        return F.pad(x, (0, 0, pad_len, 0), value=pad_value) if x.dim() == 2 else F.pad(x, (pad_len, 0), value=pad_value)
                    else:
                        return F.pad(x, (0, 0, 0, pad_len), value=pad_value) if x.dim() == 2 else F.pad(x, (0, pad_len), value=pad_value)
                else:
                    if tokenizer.padding_side == 'left':
                        return [pad_value] * pad_len + x
                    else:
                        return x + [pad_value] * pad_len
            return x

        batch = {}
        batch["input_ids"] = torch.stack([pad_and_trunc(f["input_ids"], max_len, tokenizer.pad_token_id) for f in features])
        batch["attention_mask"] = torch.stack([pad_and_trunc(f["attention_mask"], max_len, 0) for f in features])
        batch["labels"] = torch.stack([pad_and_trunc(f["labels"], max_len, -100) for f in features])
        batch["topk_indices"]  = torch.stack([pad_and_trunc(f["topk_indices"], max_len, 0) for f in features])
        batch["topk_values"]   = torch.stack([pad_and_trunc(f["topk_values"], max_len, 0.0) for f in features])
        if "teacher_mask" in features[0]:
            batch["teacher_mask"] = torch.stack([pad_and_trunc(f["teacher_mask"], max_len, 0) for f in features])
        if "next_ids" in features[0]:
            batch["next_ids"]     = torch.stack([pad_and_trunc(f["next_ids"], max_len, -100) for f in features])
        return batch

    max_length = config.get('max_seq_length', 8192)

    train_loader = None
    train_ds = None
    train_cfg = config.get('train', {})
    if train_cfg.get('data_path') and train_cfg.get('teacher_logits'):
        train_ds = DistillationDataset(
            data_path=train_cfg['data_path'],
            teacher_logits_path=train_cfg['teacher_logits'],
            tokenizer=tokenizer,
            max_length=max_length,
            only_target_loss=config.get('only_target_loss')
        )
        #train_loader = DataLoader(
        #    train_ds,
        #    batch_size=trainer_cfg.get('per_device_train_batch_size', 16),
        #    collate_fn=collate_fn,
        #    num_workers=args.num_workers,
        #    pin_memory=True
        #)
    if is_main_process():
        for idx, rec in enumerate(train_ds):
            print(f"Sample {idx} keys: {list(rec.keys())}")
            if idx >= 5:
                break

    val_loader = None
    val_ds = None
    validation_cfg = config.get('validation', {})
    if validation_cfg and validation_cfg.get('data_path') and validation_cfg.get('teacher_logits'):
        val_ds = DistillationDataset(
            data_path=validation_cfg['data_path'],
            teacher_logits_path=validation_cfg['teacher_logits'],
            tokenizer=tokenizer,
            max_length=max_length,
            only_target_loss=config.get('only_target_loss')
        )
        #val_loader = DataLoader(
        #    val_ds,
        #    batch_size=trainer_cfg.get('per_device_eval_batch_size', 16),
        #    collate_fn=collate_fn,
        #    num_workers=args.num_workers,
        #    pin_memory=True
        #)
    
    dist.barrier() 

    if is_main_process():
         for idx, rec in enumerate(val_ds):
            print(f"Val sample {idx} keys: {list(rec.keys())}")
            if idx >= 5:
                break
    
    
    if is_main_process():
        dbg_loader = DataLoader(train_ds, batch_size=2, collate_fn=collate_fn)
        dbg_batch = next(iter(dbg_loader))

        batch_size = dbg_batch['input_ids'].size(0)
        for i in range(batch_size):
            print(f"=== DEBUG SAMPLE {i} ===")
            inp      = dbg_batch['input_ids'][i]
            attn     = dbg_batch['attention_mask'][i]
            labels_tensor = dbg_batch.get('labels', None)
            labels   = labels_tensor[i] if labels_tensor is not None else None
            topk_idx = dbg_batch['topk_indices'][i]
            topk_val = dbg_batch['topk_values'][i]

            print(f"=== topk shift check ===")

            print(f"{'t':>2}  {'inp_id':>6}  {'inp_tok':>10}   {'label_id':>8}  {'label_tok':>10}   {'mask':>4}")
            print("-" * 60)

            for t in range(28, 48):
                inp_id = int(inp[t].item())
                lab_id = int(labels[t].item())
                inp_tok = tokenizer.decode([inp_id]).replace("\n","\\n")
                lab_tok = tokenizer.decode([lab_id]) if lab_id != -100 else "<pad>"
                m = int(attn[t].item())
                print(f"{t:2d}   {inp_id:6d}   {inp_tok:10s}   {lab_id:8d}   {lab_tok:10s}   {m:4d}")



            print("=== RAW DEBUG SAMPLE ===")
            print("Decoded tail (last 150 tokens):")
            print(tokenizer.decode(inp))
            print(inp[-5:])
            print("Shapes: input_ids", inp.shape, " topk_indices", topk_idx.shape, " topk_values", topk_val.shape)

            print("Raw teacher topk_indices (first 20 positions):")
            for t in range(20):
                print(f" sample {i} pos {t:2d}: {topk_idx[t].tolist()}")

            # student logits
            student.eval()
            device = student.device
            inp_batch  = inp.unsqueeze(0).to(device)
            attn_batch = attn.unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = student(input_ids=inp_batch, attention_mask=attn_batch)
                stud_logits = outputs.logits[0]

            print("Student logits at teacher's top-K positions for sample 0, first 5 positions:")
            topk = torch.topk(stud_logits, k=5, dim=-1)
            indices = topk.indices.cpu().tolist()
            values = topk.values.cpu().tolist()
            for t in range(20):
                print(f" pos {t:2d}: {tokenizer.decode(indices[t])}")

            for t in range(20):
                print(f" pos {t:2d}: {topk_val[t]}")

            for t in range(20):
                idxs = topk_idx[t].tolist()
                vals = [stud_logits[t, idx].item() for idx in idxs]
                print(f" pos {t:2d}: {vals}")
        """
        pred_top1 = topk_idx[:-1, 0]
        next_t = inp[1:1+pred_top1.size(0)]
        mask_valid = attn[1:1+pred_top1.size(0)].bool()
        mismatch_next = (pred_top1 != next_t)[mask_valid].float().mean().item() if mask_valid.any() else 0.0

        pred_top1_same = topk_idx[:, 0]
        mismatch_same = (pred_top1_same != inp)[attn.bool()].float().mean().item() if attn.bool().any() else 0.0

        print(f"Mismatch rate (next-token): {mismatch_next:.4f}")
        print(f"Mismatch rate (same-token): {mismatch_same:.4f}")

        first_real = int((attn == 1).nonzero(as_tuple=True)[0][0])
        print_token_rows(inp, labels, attn, topk_idx, topk_val, tokenizer, start=first_real, end=first_real+25)
    
        print('=== DEBUG SELECTED RANGE ===')
        print("Detailed logits and token info for positions 7435–7459:")
        
        for t in range(7435, 7460):
            inp_id = int(inp[t])
            token_str = tokenizer.decode([inp_id])
            teacher_ids = topk_idx[t].tolist()
            teacher_tokens = [tokenizer.decode([tid]) for tid in teacher_ids]
            teacher_logits = topk_val[t].tolist()
            student_vals = [stud_logits[t, idx].item() for idx in teacher_ids]
            print(f"pos {t}: inp_id={inp_id} tok='{token_str}'")
            print(f"      teacher_ids: {teacher_ids}")
            print(f"      teacher tokens: {teacher_tokens}")
            print(f"      teacher_logits: {[round(v,4) for v in teacher_logits]}")
            print(f"      student_logits: {[round(v,4) for v in student_vals]}")
        
        print("=== END RAW DEBUG SAMPLE ===")
    """
    
    #train_sample = Subset(train_ds, [0])
    #val_sample = Subset(val_ds, [0])

    dist.barrier()

    trainer = DistillTrainer(
        model=student,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        per_token=config.get('per_token', True),
        loss_multi=config.get('loss_multi', 0.1)
    )


    trainer.train()

    student.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    """
    if is_main_process():
        rec = train_sample[0]

        raw_idx = rec['_idx']
        raw_rec = train_ds.data_records[raw_idx]

        prompt_msgs = [{'role': m['role'], 'content': m['content']}
                        for m in raw_rec['messages'] if m.get('role') != 'assistant' and 'surface' not in m]

        prompt_ids = tokenizer.apply_chat_template(
            prompt_msgs, 
            add_special_tokens=False, 
            tokenize=True, 
            add_generation_prompt=True
        )

        device = student.device
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).to(device)

        output = student.generate(prompt_tensor, max_new_token=2000, return_dict_in_generate=True, output_scores=True)
        gen_seq = output.sequences[0, len(prompt_ids):]
        print("===Generated Output===")
        print(tokenizer.decode(gen_seq))

        with torch.no_grad():
            full_output = student(input_ids=rec['input_ids'].unsqueeze(0).to(device), attention_mask=rec['attention_mask'].unsqueeze(0).to(device))
            logits = full_output.logits
            teacher_idx = rec['topk_indices']
            teacher_vals = rec['topk_values']
            student_topk = logits.gather(dim=-1, index=teacher_idx.to(device).cpu())

        print("===Logits comparsion===")
        for i in range(student_topk.size(0)):
            print(f"pos {i:3d}: student {student_topk[i].tolist()}, teacher {teacher_vals[i].tolist()}")
        """

        #student_dir = os.path.join(args.output_dir, "student")
    


if __name__ == "__main__":
    main()