import random
import json
import os

import fire
#import wandb
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForTokenClassification,
    AutoConfig,
)
from transformers import (
    Trainer,
    TrainingArguments,
    logging,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    BitsAndBytesConfig,
    HfArgumentParser
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from peft import get_peft_model, LoraConfig
import re
#from unsloth.models._utils import prepare_model_for_kbit_training
from peft import prepare_model_for_kbit_training
#from .utils import prepare_model_for_kbit_training
from .dataset import ChatDataset
from .utils import set_random_seed
from .utils import read_jsonl
import codecs

os.environ["WANDB_DISABLED"] = "true"

def train(
    config_file: str,
    train_file: str,
    val_file: str,
    output_dir: str,
    custom_chat_template_path: str = None,
    sample_rate: float = 1.0,
    seed: int = 42,
):
    #set_random_seed(seed)
    #logging.set_verbosity_info()
    print(os.getenv('CUDA_VISIBLE_DEVICES', 'none'))
    print(custom_chat_template_path)
    with open(config_file, "r") as r:
        config = json.load(r)
    

    trainer_config = config.get("trainer")
    trainer_config['output_dir'] = output_dir
    parser = HfArgumentParser((TrainingArguments))
    training_args = parser.parse_dict(trainer_config)[0]
    #training_args.output_dir = output_dir
    lora_config = config.get("lora")
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    #trainer_config['device'] = f"cuda:{local_rank}"
    #training_args = TrainingArguments(
    #    output_dir=output_dir, **trainer_config
    #)
    #training_args.device = f"cuda:{training_args.local_rank}"
    print(training_args)
    print(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    #training_args.device = f"cuda:{local_rank}"
    print(
        f"device: {training_args.device}"
    )
    print(training_args.distributed_state)
    print(training_args.distributed_state.device)
    print(
        f"device: {training_args.device}"
    )
    print('LOCAL RANK: ', local_rank)
    #exit(1)
    model_name = config["model_name"]
    tokenizer_name = config.get("tokenizer_name", model_name)

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

    #tokenizer.add_special_tokens({'pad_token': config["pad_token"]})
    #tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    #tokenizer.save_pretrained(output_dir)

    print(config)

    train_records = read_jsonl(train_file)
    val_records = read_jsonl(val_file)
    random.shuffle(train_records)
    print(train_records[0])

    only_target_loss = config.get("only_target_loss", True)
    max_tokens_count = config["max_tokens_count"]
    print(max_tokens_count)
    datasets = []
    for records in (train_records, val_records):
        datasets.append(
            ChatDataset(
                records,
                tokenizer,
                max_tokens_count=max_tokens_count,
                sample_rate=sample_rate,
                only_target_loss=only_target_loss,
                add_global_eos=False,
                add_global_bos=False
            )
        )
    train_dataset, val_dataset = datasets
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    print("INPUT_IDS")
    print(data_collator([train_dataset[0], train_dataset[1]])["input_ids"][0])
    print("MASK")
    print(data_collator([train_dataset[0], train_dataset[1]])["attention_mask"][0])
    print("LABELS")
    print(data_collator([train_dataset[0], train_dataset[1]])["labels"][0])

    print('train_max ', max([len(t['input_ids']) for t in train_dataset]))
    print('val_max ', max([len(t['input_ids']) for t in val_dataset]))
    print(len(train_dataset))
    print(len(val_dataset))
    load_in_8bit = bool(config.get("load_in_8bit", False))
    load_in_4bit = bool(config.get("load_in_4bit", False))
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
    #model = prepare_model_for_kbit_training(model)
    
    gradient_checkpointing = config.get('gradient_checkpointing', False)
    if load_in_4bit or load_in_8bit:
        print('prepare')
        #prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": True})
        prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    else:
        if gradient_checkpointing:
            #model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

    if lora_config:
        lora_config = LoraConfig(**lora_config)
        model = get_peft_model(model, lora_config)
        if model.config.tie_word_embeddings and 'lm_head' in config['lora']['modules_to_save']:
            print('Tie embeddings')
            print(model)
            model.base_model.model.model.embed_tokens.weight = model.base_model.model.lm_head.modules_to_save["default"].weight

    else:
        for param_name, param in model.model.named_parameters():
            if 'embed_tokens' not in param_name:
                param.requires_grad = False
    training_args.output_dir = output_dir
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    if len(trainer.label_names) == 0:
        trainer.label_names.append('labels')

    trainer.train()

    #print(model.model.model.embed_tokens.weight[0])
    #print(model.lm_head.original_module.weight[0])
    #print(model.lm_head.modules_to_save['default'].weight[0])
    #model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    trainer.save_model()

if __name__ == "__main__":
    fire.Fire(train)
