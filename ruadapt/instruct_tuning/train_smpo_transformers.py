import json
import random
import fire
from typing import List, Dict

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
from datasets import Dataset as HFDataset

from .smpo_trainer import SimpleMarginPOConfig, SimpleMarginPOTrainer
from .dpo_dataset import DPODataset
from .utils import read_jsonl
import os

import json
import random
import fire
from typing import List, Dict

#import wandb
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
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from trl import CPOConfig, CPOTrainer
from datasets import Dataset as HFDataset
#from unsloth import PatchDPOTrainer, FastLanguageModel
from peft import prepare_model_for_kbit_training
#from .utils import prepare_model_for_kbit_training
from .utils import read_jsonl
import os
import codecs

def train(
    config_file: str,
    train_path: str,
    eval_path: str,
    output_dir: str,
    custom_chat_template_path: str | None = None,
    sample_rate: float = 1.0,
):
    #PatchDPOTrainer()
    with open(config_file, "r") as r:
        config = json.load(r)

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    print('LOCAL RANK: ', local_rank)
    max_tokens_count = config["max_tokens_count"]
    max_seq_length = config.get("max_seq_length", max_tokens_count)
    model_name = config["model_name"]

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
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=f"cuda:{local_rank}",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = config["pad_token"]
    tokenizer.eos_token = config["eos_token"]
    tokenizer.bos_token = config["bos_token"]
    tokenizer.padding_side = "left"

    if custom_chat_template_path is not None:
        with codecs.open(custom_chat_template_path, 'r', 'utf-8') as file:
            tokenizer.chat_template = json.load(file)

    tokenizer.save_pretrained(output_dir)

    gradient_checkpointing = config.get('gradient_checkpointing', False)
    if config["load_in_4bit"] or config["load_in_8bit"]:
        print('prepare')
        #prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": True})
        prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    else:
        if gradient_checkpointing:
            #model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

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
    train_dataset = DPODataset(
        train_records,
        tokenizer=tokenizer,
        max_tokens_count=max_tokens_count,
        sample_rate=sample_rate,
        apply_chat_template=True,
    )
    train_dataset = HFDataset.from_list(train_dataset)
    eval_records = read_jsonl(eval_path)
    eval_dataset = DPODataset(
        eval_records,
        tokenizer=tokenizer,
        max_tokens_count=max_tokens_count,
        sample_rate=sample_rate,
        apply_chat_template=True,
    )
    eval_dataset = HFDataset.from_list(eval_dataset)
    print(train_dataset[0])

    trainer_config = config.get("trainer")
    #if trainer_config.get("report_to", "wandb") == "wandb":
    #    wandb.init(project="rulm_self_instruct", name=config_file)

    training_args = SimpleMarginPOConfig(
        output_dir=output_dir, report_to="tensorboard", **config["smpo"], **trainer_config
    )

    trainer = SimpleMarginPOTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    fire.Fire(train)