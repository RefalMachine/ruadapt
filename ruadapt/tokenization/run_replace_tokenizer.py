from torch import nn
from tqdm import tqdm
import torch
import argparse
from transformers import AutoTokenizer, LlamaTokenizer, AutoConfig, AutoModel
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
import torch
from .replace_tokenizer import reinit_embeddings_with_head_universal
from .utils import special_encode
import codecs
import json
import os
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path')
    parser.add_argument('--new_tokenizer_path')
    parser.add_argument('--output_path')
    parser.add_argument('--mode', default='mean')
    parser.add_argument('--mult', default=1.0, type=float)
    args = parser.parse_args()
    print(args)
    
    

    tokenizer_old = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer_new = AutoTokenizer.from_pretrained(args.new_tokenizer_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    
    # Динамический выбор класса архитектуры
    architectures = getattr(config, "architectures", [])
    if architectures and "Qwen3_5ForConditionalGeneration" in architectures:
        print("Detected VLM architecture. Loading via Qwen3_5ForConditionalGeneration.")
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
        ModelClass = Qwen3_5ForConditionalGeneration
    else:
        print(f"Detected Text architecture: {architectures}. Loading via AutoModelForCausalLM.")
        from transformers import AutoModelForCausalLM
        ModelClass = AutoModelForCausalLM

    model = ModelClass.from_pretrained(
        args.model_name_or_path, 
        device_map='cuda:0',
        torch_dtype=config.torch_dtype,
        trust_remote_code=True
    )
    print(model)
    # Извлечение матриц согласно распечатке структуры модели:
    # model (Qwen3_5ForConditionalGeneration)
    #  -> .model (Qwen3_5Model)
    #      -> .language_model (Qwen3_5TextModel)
    #          -> .embed_tokens
    #  -> .lm_head
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        embed_tokens = model.model.language_model.embed_tokens
        lm_head = model.lm_head
    elif hasattr(model, 'language_model'):
        embed_tokens = model.language_model.model.embed_tokens if hasattr(model.language_model, 'model') else model.language_model.embed_tokens
        lm_head = getattr(model, 'lm_head', getattr(model.language_model, 'lm_head', None))
    else:
        embed_tokens = model.model.embed_tokens
        lm_head = model.lm_head

    embeddings_old = embed_tokens.weight.data.clone()
    lm_head_old = lm_head.weight.data.clone()

    target_multiple = 256
    current_len = len(tokenizer_new)
    pad_needed = (target_multiple - (current_len % target_multiple)) % target_multiple
    
    # СНАЧАЛА добавляем токены и ресайзим матрицы
    if pad_needed > 0:
        print(f"Padding vocabulary from {current_len} to {current_len + pad_needed} (+{pad_needed} free tokens)")
        new_tokens = [f"<|free_token{i+1}|>" for i in range(pad_needed)]
        tokenizer_new.add_tokens(new_tokens)
        model.resize_token_embeddings(len(tokenizer_new))

    # Переинициализация весов
    reinit_logs = reinit_embeddings_with_head_universal(model, tokenizer_old, tokenizer_new, mode=args.mode, lm_head_init='hm', mult=args.mult)

    # Обновление конфига
    model.config.bos_token_id = tokenizer_new.bos_token_id
    model.config.eos_token_id = tokenizer_new.eos_token_id
    model.config.pad_token_id = tokenizer_new.pad_token_id
    model.generation_config.bos_token_id = tokenizer_new.bos_token_id
    model.generation_config.eos_token_id = tokenizer_new.eos_token_id
    model.generation_config.pad_token_id = tokenizer_new.pad_token_id
    
    model.config.vocab_size = len(tokenizer_new)

    print("Saving model and tokenizer...")
    model.save_pretrained(args.output_path)
    tokenizer_new.save_pretrained(args.output_path)

    with codecs.open(os.path.join(args.output_path, 'reinit_tokenizer_logs.json'), 'w', 'utf-8') as file:
        json.dump(reinit_logs, file, ensure_ascii=False, indent=4)
        
    print("Done. VLM Architecture preserved.")
