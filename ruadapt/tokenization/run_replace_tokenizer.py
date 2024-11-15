from torch import nn
from tqdm import tqdm
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, AutoConfig
import torch
from .replace_tokenizer import reinit_embeddings_with_head_universal
from .utils import special_encode
import codecs
import json
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path')
    parser.add_argument('--new_tokenizer_path')
    parser.add_argument('--output_path')
    #parser.add_argument('--type', default='mistral')
    args = parser.parse_args()

    tokenizer_old = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer_new = AutoTokenizer.from_pretrained(args.new_tokenizer_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map='cuda:0', torch_dtype=config.torch_dtype)

    embeddings_old = model.model.embed_tokens.weight.data.clone()
    lm_head_old = model.lm_head.weight.data.clone()

    reinit_logs = reinit_embeddings_with_head_universal(model, tokenizer_old, tokenizer_new, mode='mean', lm_head_init='hm')

    model.config.bos_token_id = tokenizer_new.bos_token_id
    model.config.eos_token_id = tokenizer_new.eos_token_id
    model.config.pad_token_id = tokenizer_new.pad_token_id
    model.generation_config.bos_token_id = tokenizer_new.bos_token_id
    model.generation_config.eos_token_id = tokenizer_new.eos_token_id
    model.generation_config.pad_token_id = tokenizer_new.pad_token_id
    
    word = 'онный'
    print(word)
    #new_id = tokenizer_new(word, add_special_tokens=False)['input_ids']
    new_id = special_encode(word, tokenizer_new)
    old_ids = special_encode(word, tokenizer_old)
    print(new_id, old_ids)

    print('Embeddings')
    print(embeddings_old[old_ids])
    print(embeddings_old[old_ids].mean(axis=0))
    print(model.model.embed_tokens.weight[new_id])

    print('LM head')
    print(lm_head_old[old_ids])
    print(lm_head_old[old_ids].mean(axis=0))
    print(model.lm_head.weight[[new_id]])

    word = ' Пушкин'
    print(word)
    #new_id = tokenizer_new(word, add_special_tokens=False)['input_ids']
    new_id = special_encode(word, tokenizer_new)
    old_ids = special_encode(word, tokenizer_old)
    print(new_id, old_ids)

    print('Embeddings')
    print(embeddings_old[old_ids])
    print(embeddings_old[old_ids].mean(axis=0))
    print(model.model.embed_tokens.weight[new_id])

    print('LM head')
    print(lm_head_old[old_ids])
    print(lm_head_old[old_ids].mean(axis=0))
    print(model.lm_head.weight[[new_id]])


    word = '\n'
    print('|' + word + '|')
    #new_id = tokenizer_new(word, add_special_tokens=False)['input_ids']
    new_id = special_encode(word, tokenizer_new)
    old_ids = special_encode(word, tokenizer_old)
    print(new_id, old_ids)

    print('Embeddings')
    print(embeddings_old[old_ids])
    print(embeddings_old[old_ids].mean(axis=0))
    print(model.model.embed_tokens.weight[new_id])

    print('LM head')
    print(lm_head_old[old_ids])
    print(lm_head_old[old_ids].mean(axis=0))
    print(model.lm_head.weight[[new_id]])

    word = tokenizer_old.bos_token
    if word is None:
        print(tokenizer_old)
        print(tokenizer_old.special_tokens_map)
        word = tokenizer_old.special_tokens_map.get('bos_token', None)
    if word is not None:
        print('|' + word + '|')
        #new_id = tokenizer_new(word, add_special_tokens=False)['input_ids']
        new_id = special_encode(word, tokenizer_new)
        old_ids = special_encode(word, tokenizer_old)
        print(new_id, old_ids)

        print('Embeddings')
        print(embeddings_old[old_ids])
        print(embeddings_old[old_ids].mean(axis=0))
        print(model.model.embed_tokens.weight[new_id])

        print('LM head')
        print(lm_head_old[old_ids])
        print(lm_head_old[old_ids].mean(axis=0))
        print(model.lm_head.weight[[new_id]])

    word = tokenizer_old.eos_token
    if word is None:
        print(tokenizer_old)
        print(tokenizer_old.special_tokens_map)
        word = tokenizer_old.special_tokens_map['eos_token']
    if word is not None:
        print('|' + word + '|')
        #new_id = tokenizer_new(word, add_special_tokens=False)['input_ids']
        new_id = special_encode(word, tokenizer_new)
        old_ids = special_encode(word, tokenizer_old)
        print(new_id, old_ids)

        print('Embeddings')
        print(embeddings_old[old_ids])
        print(embeddings_old[old_ids].mean(axis=0))
        print(model.model.embed_tokens.weight[new_id])

        print('LM head')
        print(lm_head_old[old_ids])
        print(lm_head_old[old_ids].mean(axis=0))
        print(model.lm_head.weight[[new_id]])

    word = '\n\n'
    print('|' + word + '|')
    #new_id = tokenizer_new(word, add_special_tokens=False)['input_ids']
    new_id = special_encode(word, tokenizer_new)
    old_ids = special_encode(word, tokenizer_old)
    print(new_id, old_ids)

    print('Embeddings')
    print(embeddings_old[old_ids])
    print(embeddings_old[old_ids].mean(axis=0))
    print(model.model.embed_tokens.weight[new_id])

    print('LM head')
    print(lm_head_old[old_ids])
    print(lm_head_old[old_ids].mean(axis=0))
    print(model.lm_head.weight[[new_id]])

    word = '\t'
    print('|' + word + '|')
    #new_id = tokenizer_new(word, add_special_tokens=False)['input_ids']
    new_id = special_encode(word, tokenizer_new)
    old_ids = special_encode(word, tokenizer_old)
    print(new_id, old_ids)

    print('Embeddings')
    print(embeddings_old[old_ids])
    print(embeddings_old[old_ids].mean(axis=0))
    print(model.model.embed_tokens.weight[new_id])

    print('LM head')
    print(lm_head_old[old_ids])
    print(lm_head_old[old_ids].mean(axis=0))
    print(model.lm_head.weight[[new_id]])

    word = ' '
    print('|' + word + '|')
    #new_id = tokenizer_new(word, add_special_tokens=False)['input_ids']
    new_id = special_encode(word, tokenizer_new)
    old_ids = special_encode(word, tokenizer_old)
    print(new_id, old_ids)

    print('Embeddings')
    print(embeddings_old[old_ids])
    print(embeddings_old[old_ids].mean(axis=0))
    print(model.model.embed_tokens.weight[new_id])

    print('LM head')
    print(lm_head_old[old_ids])
    print(lm_head_old[old_ids].mean(axis=0))
    print(model.lm_head.weight[[new_id]])

    #print(embeddings_old[[128000, 128001, 128005]])
    #print(model.model.embed_tokens.weight[[128000, 128001, 128005]])
    #print(model.model.embed_tokens.weight[[32000, 32001, 32005]])
    #print(lm_head_old[[128000, 128001, 128005]])
    #print(model.lm_head.weight[[128000, 128001, 128005]])
    #print(model.lm_head.weight[[32000, 32001, 32005]])
    #print(model.model.embed_tokens.weight)
    #print(model.lm_head.weight)

    new_tokens = [f"<|free_token{i+1}|>" for i in range(21)]
    tokenizer_new.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer_new))
    print(model.lm_head.weight.shape)
    model.save_pretrained(args.output_path)
    tokenizer_new.save_pretrained(args.output_path)

    with codecs.open(os.path.join(args.output_path, 'reinit_tokenizer_logs.json'), 'w', 'utf-8') as file:
        json.dump(reinit_logs, file, ensure_ascii=False, indent=4)
