import json
import random
import fire
from typing import List, Dict

import wandb
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from trl import CPOConfig, CPOTrainer
from datasets import Dataset as HFDataset
#from unsloth import PatchDPOTrainer, FastLanguageModel

from .utils import read_jsonl
import os

os.environ["WANDB_DISABLED"] = "true"

class ChatCPODataset(Dataset):
    def __init__(
        self,
        original_records: List[Dict],
        tokenizer: AutoTokenizer,
        max_tokens_count: int,
        sample_rate: float = 1.0,
    ):
        self.original_records = original_records
        self.tokenizer = tokenizer
        self.max_tokens_count = max_tokens_count
        self.sample_rate = sample_rate

        self.records = []
        for record in tqdm(original_records):
            if random.random() > self.sample_rate:
                continue

            prompt_messages = record["prompt"]
            prompt = self.tokenizer.apply_chat_template(
                prompt_messages, add_generation_prompt=True, tokenize=False
            )
            prompt = prompt.replace(self.tokenizer.bos_token, "")

            prompt_tokens = self.tokenizer.apply_chat_template(
                prompt_messages, add_generation_prompt=True, tokenize=True
            )
            chosen = record["chosen"][0]["content"]
            chosen_tokens = self.tokenizer(chosen)["input_ids"]

            rejected = record["rejected"][0]["content"]
            rejected_tokens = self.tokenizer(rejected)["input_ids"]

            if len(prompt_tokens) + len(chosen_tokens) > self.max_tokens_count - 10:
                continue
            if len(prompt_tokens) + len(rejected_tokens) > self.max_tokens_count - 10:
                continue

            self.records.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def train(
    config_file: str,
    train_path: str,
    eval_path: str,
    output_dir: str,
    sample_rate: float = 1.0,
):
    #PatchDPOTrainer()
    with open(config_file, "r") as r:
        config = json.load(r)

    max_tokens_count = config["max_tokens_count"]
    max_seq_length = config.get("max_seq_length", max_tokens_count)
    model_name = config["model_name"]
    '''
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_8bit=config["load_in_8bit"],
        load_in_4bit=config["load_in_4bit"],
        attn_implementation="flash_attention_2",
    )'''
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=config["load_in_8bit"],
        load_in_4bit=config["load_in_4bit"],
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = config["pad_token"]
    tokenizer.eos_token = config["eos_token"]
    tokenizer.bos_token = config["bos_token"]
    tokenizer.padding_side = "left"
    tokenizer.save_pretrained(output_dir)
    gradient_checkpointing = config.get('gradient_checkpointing', False)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_config = config["lora"]
    if lora_config:
        lora_config = LoraConfig(**lora_config)
        model = get_peft_model(model, lora_config)
        if model.config.tie_word_embeddings and 'lm_head' in config['lora'].get('modules_to_save', []):
            assert 'lm_head' not in config['lora']['modules_to_save'] or 'embed_tokens' not in config['lora']['modules_to_save']
            print('Tie embeddings')
            print(model)
            model.base_model.model.model.embed_tokens.weight = model.base_model.model.lm_head.modules_to_save["default"].weight

    train_records = read_jsonl(train_path)
    train_dataset = ChatCPODataset(
        train_records,
        tokenizer=tokenizer,
        max_tokens_count=max_tokens_count,
        sample_rate=sample_rate,
    )
    train_dataset = HFDataset.from_list(train_dataset)
    eval_records = read_jsonl(eval_path)
    eval_dataset = ChatCPODataset(
        eval_records,
        tokenizer=tokenizer,
        max_tokens_count=max_tokens_count,
        sample_rate=sample_rate,
    )
    eval_dataset = HFDataset.from_list(eval_dataset)
    print(train_dataset[0])

    trainer_config = config.get("trainer")
    #if trainer_config.get("report_to", "wandb") == "wandb":
    #    wandb.init(project="ruadapt", name=config_file)

    training_args = CPOConfig(
        output_dir=output_dir, **config["cpo"], **trainer_config
    )

    trainer = CPOTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)