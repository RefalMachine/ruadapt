from torch import nn
from tqdm import tqdm
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, AutoConfig
import torch
#from replace_tokenizer import reinit_embeddings_with_head_universal
#from utils import special_encode
import codecs
import json
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path1')
    parser.add_argument('--model_name_or_path2')
    args = parser.parse_args()

    tokenizer1 = AutoTokenizer.from_pretrained(args.model_name_or_path1)
    tokenizer2 = AutoTokenizer.from_pretrained(args.model_name_or_path2)

    config1 = AutoConfig.from_pretrained(args.model_name_or_path1)
    model1 = AutoModelForCausalLM.from_pretrained(args.model_name_or_path1, device_map='cuda:0', torch_dtype=config1.torch_dtype)

    config2 = AutoConfig.from_pretrained(args.model_name_or_path2)
    model2 = AutoModelForCausalLM.from_pretrained(args.model_name_or_path2, device_map='cuda:2', torch_dtype=config2.torch_dtype)

    embeddings1 = model1.model.embed_tokens.weight.data.clone().cpu()
    lm_head1 = model1.lm_head.weight.data.clone().cpu()

    embeddings2 = model2.model.embed_tokens.weight.data.clone().cpu()
    lm_head2 = model2.lm_head.weight.data.clone().cpu()

    common_tokens = set(tokenizer1.vocab).intersection(set(tokenizer2.vocab))
    print(len(common_tokens), len(tokenizer1.vocab), len(tokenizer2.vocab))
    print([token for token in tokenizer1.vocab if token not in common_tokens])
    print([token for token in tokenizer2.vocab if token not in common_tokens])

    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert lm_head1.shape[0] == lm_head2.shape[0]

    count = 0
    #print(embeddings1[1999])
    #print(embeddings2[1999])
    logs = []
    for i in tqdm(range(embeddings1.shape[0])):
        diff = embeddings1[i] - embeddings2[i]
        if diff.norm() > 1e-12:
            logs.append(['input', i, tokenizer1.convert_ids_to_tokens([i]), tokenizer2.convert_ids_to_tokens([i]), float(diff.norm())])
            '''
            print(i)
            print(embeddings1[i])
            print(embeddings2[i])
            print(tokenizer1.convert_ids_to_tokens([i]))
            print(tokenizer2.convert_ids_to_tokens([i]))
            print(diff.norm())
            print()
            '''
            count += 1
    #print(diff.shape)
    #print(diff)
    print('emb_diff count ' + str(count))

    count = 0
    for i in tqdm(range(lm_head1.shape[0])):
        diff = lm_head1[i] - lm_head2[i]
        if diff.norm() > 1e-12:
            logs.append(['output', i, tokenizer1.convert_ids_to_tokens([i]), tokenizer2.convert_ids_to_tokens([i]), float(diff.norm())])
            '''
            print(i)
            print(lm_head1[i])
            print(lm_head2[i])
            print(tokenizer1.convert_ids_to_tokens([i]))
            print(tokenizer2.convert_ids_to_tokens([i]))
            print(diff.norm())
            print()
            '''
            count += 1
    #print(diff.shape)
    #print(diff)
    print('lh_diff count ' + str(count))
    with codecs.open('test_logs.json', 'w', 'utf-8') as file:
        json.dump(logs, file, ensure_ascii=False, indent=4)