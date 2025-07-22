#!/usr/bin/env python3
"""
Скрипт для обучения SFT с использованием FSDP (Fully Sharded Data Parallel)
Поддерживает обучение моделей 32B+ параметров на 8 GPU в рамках одного узла
"""

import random
import json
import os
import functools
from datasets import load_dataset
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


def convert_single_record(
    example: Dict,
    tokenizer: Dict = None,
    max_tokens_count: int = 2048,
    only_target_loss: bool = True,
    add_global_bos: bool = True,
    add_global_eos: bool = True,
    labels_pad_token_id: int = -100,
    sample_rate: float = 1.0
) -> Dict:
    """
    Функция для обработки одной записи через datasets.map().
    Возвращает обработанную запись или помечает для фильтрации.
    """
    
    # Применяем sample_rate
    skip_return = {
        "input_ids": torch.LongTensor([]),
        "labels": torch.LongTensor([]),
        "attention_mask": torch.LongTensor([]),
        "skip": True,
        "len": 0
    }
    if random.random() > sample_rate:
        return skip_return
    
    # Получаем токенизатор из глобального контекста
    def get_tokens(messages):
        tokens = tokenizer.apply_chat_template(
            messages,
            add_special_tokens=False,
            tokenize=True,
            add_generation_prompt=False,
        )
        if tokens and tokens[0] == tokenizer.bos_token_id:
            tokens = tokens[1:]
        return tokens

    try:
        input_ids, labels = [], []
        
        # Нормализация ролей
        normalized_messages = []
        for message in example["messages"]:
            msg_copy = message.copy()
            if msg_copy['role'] == 'bot':
                msg_copy['role'] = 'assistant'
            normalized_messages.append(msg_copy)
        
        # Обработка сообщений с обрезанием по длине
        for mid, message in enumerate(normalized_messages):
            message_tokens = get_tokens([message])
            
            # Проверяем, помещается ли сообщение
            if len(input_ids) + len(message_tokens) > max_tokens_count - 2:
                break
                
            # Создаем маски для labels
            if message["role"] in ("assistant", "bot", "gpt") or not only_target_loss:
                message_labels = message_tokens.copy()
                if 'mask_think_block' in example and example['mask_think_block'] and (mid == len(normalized_messages) - 1):
                    prefix = '<|im_start|>assistant\n<think>\n\n</think>\n\n'
                    prefix_tokens = tokenizer(prefix)['input_ids']
                    assert prefix_tokens == message_labels[:len(prefix_tokens)]
                    for i in range(len(prefix_tokens)):
                        message_labels[i] = labels_pad_token_id
            else:
                message_labels = [labels_pad_token_id] * len(message_tokens)
            
            input_ids.extend(message_tokens)
            labels.extend(message_labels)
        
        if not input_ids:
            return skip_return
        
        # Проверка эквивалентности токенизации/детокенизации
        original_tokens = get_tokens(normalized_messages)
        if input_ids != original_tokens[:len(input_ids)]:
            # Если не совпадает, пропускаем запись
            return skip_return
        
        # Добавление глобальных токенов
        if add_global_bos and input_ids[0] != tokenizer.bos_token_id:
            input_ids.insert(0, tokenizer.bos_token_id)
            labels.insert(0, labels_pad_token_id)
        
        # Удаление дублирующего EOS если есть
        if len(input_ids) >= 2 and input_ids[-2] == tokenizer.eos_token_id:
            input_ids = input_ids[:-1]
            labels = labels[:-1]
        
        if add_global_eos and input_ids[-1] != tokenizer.eos_token_id:
            input_ids.append(tokenizer.eos_token_id)
            labels.append(tokenizer.eos_token_id)
        
        # Проверяем, что есть токены для обучения
        if len([i for i in labels if i != labels_pad_token_id]) == 0:
            return skip_return
        
        # Проверяем длину
        if len(input_ids) > max_tokens_count:
            return skip_return
        
        return {
            "input_ids": torch.LongTensor(input_ids),
            "labels": torch.LongTensor(labels),
            "attention_mask": torch.LongTensor([1] * len(input_ids)),
            "len": len(input_ids),
            "skip": False
        }
        
    except Exception as e:
        return skip_return
    
class FSDPTrainer(Trainer):
    """
    Финальная версия, которая правильно обрабатывает и чекпоинты,
    и финальное сохранение модели.
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
        # --- Логика сохранения чекпоинта (остается без изменений) ---
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
            print(f"Rank 0: [FSDPTrainer] Preparing to save to {output_dir}...")
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

                print(f"Rank 0: [FSDPTrainer] Save complete for {output_dir}.")

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
    Основная функция обучения с FSDP
    
    Args:
        config_file: Путь к файлу конфигурации модели
        train_file: Путь к файлу с обучающими данными
        val_file: Путь к файлу с валидационными данными
        output_dir: Директория для сохранения результатов
        custom_chat_template_path: Путь к кастомному шаблону чата
        sample_rate: Доля данных для использования
        seed: Случайное зерно
    """
    
    # Инициализация distributed training

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    # Загрузка конфигурации
    with open(config_file, "r") as r:
        config = json.load(r)
    
    trainer_config = config.get("trainer")
    trainer_config['output_dir'] = output_dir
    
    parser = HfArgumentParser((TrainingArguments,))
    training_args, = parser.parse_dict(trainer_config)
    
    print(f"Process rank: {training_args.process_index}, device: {training_args.device}")
    print(f"World size: {training_args.world_size}")
    print(f"FSDP config from TrainingArguments: {training_args.fsdp_config}")
    print(f"Training arguments: {training_args}")
    
    lora_config = config.get("lora")
    model_name = config["model_name"]
    tokenizer_name = config.get("tokenizer_name", model_name)

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
    datasets = load_dataset('json', data_files={'train': train_file, 'validation': val_file})

    only_target_loss = config.get("only_target_loss", True)
    max_tokens_count = config["max_tokens_count"]
    
    # Определяем оптимальное количество процессов
    import multiprocessing as mp
    num_proc = min(mp.cpu_count(), 64)  # Используем больше процессов
    if training_args.process_index == 0:
        print(datasets['train'][1])
        print(datasets['train'][2])
    preprocess = lambda x: convert_single_record(x, tokenizer, max_tokens_count, only_target_loss=only_target_loss, sample_rate=sample_rate, add_global_bos=False, add_global_eos=False)
    with training_args.main_process_first(desc="chat dataset processing"):
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
        #datasets = datasets.sort("len", reverse=True)
        datasets = datasets.remove_columns(["skip", "len"])

    train_dataset, val_dataset = datasets
    train_dataset = datasets['train']
    val_dataset = datasets['validation']
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    if training_args.process_index == 0:
        print(train_dataset[1]['input_ids'])
        print(train_dataset[1]['labels'])
        print(
            "Full prompt:" + 
            tokenizer.decode(train_dataset[1]['input_ids'], skip_special_tokens=False)
        )
        print(len(train_dataset[1]['input_ids']))

        print(train_dataset[2]['input_ids'])
        print(train_dataset[2]['labels'])
        print(
            "Full prompt:" + 
            tokenizer.decode(train_dataset[2]['input_ids'], skip_special_tokens=False)
        )
        print(len(train_dataset[2]['input_ids']))
    #print(f"Max train sequence length: {max([len(t['input_ids']) for t in train_dataset])}")
    #print(f"Max val sequence length: {max([len(t['input_ids']) for t in val_dataset])}")

    
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
    TRAINER_CLASS = FSDPTrainer if training_args.fsdp_config is not None else Trainer
    trainer = TRAINER_CLASS(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    if len(trainer.label_names) == 0:
        trainer.label_names.append('labels')

    # Запуск обучения
    print("Starting training...")
    trainer.train()

    print(f"Rank {local_rank}: train() finished. Proceeding to final save.")

    # Вызываем наш переопределенный метод для сохранения финальной модели
    # Этот вызов теперь безопасен.
    trainer.save_model(output_dir)

    # Синхронизируемся после финального сохранения
    if torch.distributed.is_initialized():
        dist.barrier()

    if training_args.process_index == 0:
        print("\n=======================================================")
        print(f"TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Final model and tokenizer saved to: {output_dir}")
        print("=======================================================\n")
    
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    
    print(f"Rank {local_rank} is exiting cleanly.")

if __name__ == "__main__":
    fire.Fire(train)
