#!/usr/bin/env python3
"""
Скрипт для обучения CPO (Conservative Policy Optimization) с использованием FSDP (Fully Sharded Data Parallel)
Поддерживает обучение моделей 32B+ параметров на 8 GPU в рамках одного узла
"""

import random
import json
import os
import functools
from datasets import load_dataset, Dataset as HFDataset
import fire
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForTokenClassification
)
from transformers import (
    Trainer,
    TrainingArguments,
    HfArgumentParser
)
from peft import get_peft_model, LoraConfig
from trl import CPOConfig, CPOTrainer

# Импорты из ruadapt
from .utils import set_random_seed, read_jsonl
import codecs
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR # <--- Импортируем константу
import torch.distributed as dist
import os
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy
from typing import List, Dict, Optional
from tqdm import tqdm


def convert_single_cpo_record(
    example: Dict,
    tokenizer: AutoTokenizer = None,
    max_tokens_count: int = 2048,
    sample_rate: float = 1.0
) -> Dict:
    """
    Функция для обработки одной записи CPO через datasets.map().
    Возвращает обработанную запись или помечает для фильтрации.
    """
    
    # Применяем sample_rate
    skip_return = {
        "prompt": "",
        "chosen": "",
        "rejected": "",
        "skip": True,
        "len": 0
    }
    if random.random() > sample_rate:
        return skip_return
    
    try:
        prompt_messages = example["prompt"]
        prompt = tokenizer.apply_chat_template(
            prompt_messages, add_generation_prompt=True, tokenize=False
        )
        prompt = prompt.replace(tokenizer.bos_token, "")

        prompt_tokens = tokenizer.apply_chat_template(
            prompt_messages, add_generation_prompt=True, tokenize=True
        )
        chosen = example["chosen"][0]["content"]
        chosen_tokens = tokenizer(chosen)["input_ids"]

        rejected = example["rejected"][0]["content"]
        rejected_tokens = tokenizer(rejected)["input_ids"]

        if len(prompt_tokens) + len(chosen_tokens) > max_tokens_count - 10:
            return skip_return
        if len(prompt_tokens) + len(rejected_tokens) > max_tokens_count - 10:
            return skip_return

        return {
            "prompt": prompt, 
            "chosen": chosen, 
            "rejected": rejected,
            "skip": False,
            "len": len(prompt_tokens) + max(len(chosen_tokens), len(rejected_tokens))
        }
        
    except Exception as e:
        return skip_return


class FSDPCPOTrainer(CPOTrainer):
    """
    CPO Trainer с поддержкой FSDP и корректным сохранением моделей.
    """
    def __init__(self, *args, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        if tokenizer is not None:
            self.tokenizer = tokenizer

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Этот метод вызывается для ПРОМЕЖУТОЧНЫХ чекпоинтов.
        Сохраняем в папку checkpoint-xxx.
        """
        checkpoint_folder = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
        self._save_model_impl(checkpoint_folder)

    def save_model(self, output_dir: str = None, _internal_call: bool = False):
        """
        Этот метод вызывается для ФИНАЛЬНОГО сохранения.
        Сохраняем напрямую в output_dir.
        """
        # Убедимся, что директория указана, иначе берем из аргументов
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        
        # --- Вызываем нашу реализацию сохранения ---
        self._save_model_impl(output_dir)

    def _save_model_impl(self, output_dir: str):
        """
        Общая реализация безопасного сохранения для FSDP.
        """
        if self.args.process_index == 0:
            print(f"Rank 0: [FSDPCPOTrainer] Preparing to save to {output_dir}...")
            os.makedirs(output_dir, exist_ok=True)
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        with FSDP.summon_full_params(self.model, offload_to_cpu=True, rank0_only=True, writeback=False):
            if self.args.process_index == 0:
                print(f"Rank 0: Inside summon_full_params. Saving model to {output_dir}...")
                unwrapped_model.save_pretrained(output_dir)
                
                if self.tokenizer is not None:
                    print(f"Rank 0: Saving tokenizer to {output_dir}...")
                    self.tokenizer.save_pretrained(output_dir)

                # Сохраняем состояние тренера только для чекпоинтов
                if PREFIX_CHECKPOINT_DIR in output_dir:
                     self.save_state()

                print(f"Rank 0: [FSDPCPOTrainer] Save complete for {output_dir}.")


os.environ["WANDB_DISABLED"] = "true"

def train(
    config_file: str,
    train_file: str,
    val_file: str,
    output_dir: str,
    custom_chat_template_path: str = None,
    sample_rate: float = 1.0
):
    """
    Основная функция обучения CPO с FSDP
    
    Args:
        config_file: Путь к файлу конфигурации модели
        train_file: Путь к файлу с обучающими данными (preference data)
        val_file: Путь к файлу с валидационными данными
        output_dir: Директория для сохранения результатов
        custom_chat_template_path: Путь к кастомному шаблону чата
        sample_rate: Доля данных для использования
    """
    
    # Инициализация distributed training
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    # Загрузка конфигурации
    with open(config_file, "r") as r:
        config = json.load(r)
    
    trainer_config = config.get("trainer", {})
    cpo_config = config.get("cpo", {})
    trainer_config['output_dir'] = output_dir
    
    # Объединяем конфигурации для CPOConfig
    combined_config = {**cpo_config, **trainer_config}
    
    parser = HfArgumentParser((CPOConfig,))
    training_args, = parser.parse_dict(combined_config)
    
    print(f"Process rank: {training_args.process_index}, device: {training_args.device}")
    print(f"World size: {training_args.world_size}")
    print(f"FSDP config from TrainingArguments: {training_args.fsdp_config}")
    print(f"Training arguments: {training_args}")
    
    lora_config = config.get("lora")
    model_name = config["model_name"]
    tokenizer_name = config.get("tokenizer_name", model_name)
    max_tokens_count = config["max_tokens_count"]

    # Настройка токенизатора
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.bos_token = config["bos_token"]
    tokenizer.eos_token = config["eos_token"]
    tokenizer.pad_token = config["pad_token"]

    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids([config["bos_token"]])[0]
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids([config["eos_token"]])[0]
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids([config["pad_token"]])[0]

    if custom_chat_template_path is not None:
        with codecs.open(custom_chat_template_path, 'r', 'utf-8') as file:
            tokenizer.chat_template = json.load(file)

    tokenizer.padding_side = 'left'
    
    # Загрузка данных
    datasets = load_dataset('json', data_files={'train': train_file, 'validation': val_file})
    
    # Определяем оптимальное количество процессов
    import multiprocessing as mp
    num_proc = min(mp.cpu_count(), 64)  # Используем больше процессов
    
    if training_args.process_index == 0:
        print("Sample train record:")
        print(datasets['train'][0])
        print("Sample validation record:")
        print(datasets['validation'][0])
    
    preprocess = lambda x: convert_single_cpo_record(x, tokenizer, max_tokens_count, sample_rate=sample_rate)
    
    with training_args.main_process_first(desc="CPO dataset processing"):
        datasets = datasets.map(
            preprocess,
            batched=False,
            num_proc=num_proc,
            remove_columns=datasets['train'].column_names,
            load_from_cache_file=True
        )
        datasets = datasets.filter(
            lambda x: not x["skip"],
            num_proc=num_proc,
            load_from_cache_file=True,
            desc="Filtering valid records"
        )
        datasets = datasets.remove_columns(["skip", "len"])

    train_dataset = datasets['train']
    val_dataset = datasets['validation']

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    if training_args.process_index == 0:
        print("Processed train sample:")
        print(train_dataset[0])
        print("Processed validation sample:")
        print(val_dataset[0])
    
    # Загружаем модель без device_map для FSDP
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    
    # Настройка gradient checkpointing
    gradient_checkpointing = config.get('gradient_checkpointing', True)  # По умолчанию включено для больших моделей
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # Настройка LoRA (если используется)
    if lora_config:
        lora_config = LoraConfig(**lora_config)
        if model.config.tie_word_embeddings and 'lm_head' in config['lora']['modules_to_save']:
            if 'embed_tokens' in config['lora']['modules_to_save']:
                lora_config.modules_to_save = ['lm_head']
                config['lora']['modules_to_save'] = ['lm_head']
                print('ATTENTION!!! modules_to_save:', str(lora_config.modules_to_save))

        model = get_peft_model(model, lora_config)
        if model.config.tie_word_embeddings and 'lm_head' in config['lora']['modules_to_save']:
            model.base_model.model.model.embed_tokens.weight = model.base_model.model.lm_head.modules_to_save["default"].weight
    else:
        # Если не используем LoRA, замораживаем все параметры кроме embeddings
        for param_name, param in model.named_parameters():
            if 'embed_tokens' not in param_name:
                param.requires_grad = False

    # Создание trainer с поддержкой FSDP
    TRAINER_CLASS = FSDPCPOTrainer if training_args.fsdp_config is not None else CPOTrainer
    trainer = TRAINER_CLASS(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        tokenizer=tokenizer
    )

    # Запуск обучения
    print("Starting CPO training...")
    trainer.train()

    print(f"Rank {local_rank}: train() finished. Proceeding to final save.")

    # Вызываем наш переопределенный метод для сохранения финальной модели
    trainer.save_model(output_dir)

    # Синхронизируемся после финального сохранения
    if torch.distributed.is_initialized():
        dist.barrier()

    if training_args.process_index == 0:
        print("\n=======================================================")
        print(f"CPO TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Final model and tokenizer saved to: {output_dir}")
        print("=======================================================\n")
    
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    
    print(f"Rank {local_rank} is exiting cleanly.")

if __name__ == "__main__":
    fire.Fire(train)
