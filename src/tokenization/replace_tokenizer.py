from torch import nn
from tqdm import tqdm
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, AutoConfig
import torch
from src.tokenization.utils import reinit_embeddings_with_head_llama3, special_encode, get_mean_vec, reinit_embeddings_with_head_llama3_extended, reinit_embeddings_with_head_llama

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path')
    parser.add_argument('--new_tokenizer_path')
    parser.add_argument('--output_path')
    parser.add_argument('--type', default='default')
    args = parser.parse_args()

    tokenizer_old = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer_new = AutoTokenizer.from_pretrained(args.new_tokenizer_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map='cuda:1', torch_dtype=config.torch_dtype)

    embeddings_old = model.model.embed_tokens.weight.data.clone()
    lm_head_old = model.lm_head.weight.data.clone()

    if args.type == 'default':
        reinit_embeddings_with_head_llama(model, tokenizer_old, tokenizer_new, mode='mean', lm_head_init='hm')
    elif args.type == 'llama3':
        reinit_embeddings_with_head_llama3(model, tokenizer_old, tokenizer_new, mode='mean', lm_head_init='hm')
    elif args.type == 'llama3_ext':
        reinit_embeddings_with_head_llama3_extended(model, tokenizer_old, tokenizer_new, mode='mean', lm_head_init='hm')
    else:
        raise Exception('args.type')

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

    #print(embeddings_old[[128000, 128001, 128005]])
    #print(model.model.embed_tokens.weight[[128000, 128001, 128005]])
    #print(model.model.embed_tokens.weight[[32000, 32001, 32005]])
    #print(lm_head_old[[128000, 128001, 128005]])
    #print(model.lm_head.weight[[128000, 128001, 128005]])
    #print(model.lm_head.weight[[32000, 32001, 32005]])
    #print(model.model.embed_tokens.weight)
    #print(model.lm_head.weight)



    model.save_pretrained(args.output_path)
    tokenizer_new.save_pretrained(args.output_path)
