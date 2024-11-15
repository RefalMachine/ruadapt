from argparse import ArgumentParser
import codecs
import numpy as np
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
import os
import json
import re

byte_encoder = bytes_to_unicode()
def token_bytes_to_string(b):
    return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])

def convert_to_llama(token):
    return token_bytes_to_string(token.replace('▁', ' ').encode())

def check_ru_token(token):
    m = re.match('[ А-Яа-я]+', token)
    if m is None:
        return False
    return len(token) == len(m[0])

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--tokenizer_path')
    parser.add_argument('--output_path')
    parser.add_argument('--type')
    parser.add_argument('--only_ru', action='store_true')
    args = parser.parse_args()

    with codecs.open(os.path.join(args.tokenizer_path, 'tokenizer.json'), 'r', 'utf-8') as file:
        data = json.load(file)

    if args.type == 'unigram':
        data = data['model']['vocab']
        data = [d for d in data if d[1] < 0]
        data = [[d[0].replace('▁', ' '), str(int(1000000 * np.exp(d[1])))] for d in data]
        
        if args.only_ru:
            print('before ru filter: ', len(data))
            data = [d for d in data if check_ru_token(d[0])]
            print('after ru filter: ', len(data))

        data = ['\t'.join(d) for d in data]

    elif args.type == 'bpe':
        print('WARNING: frequency calculation may be incorrect!')

        vocab = data['model']['vocab']
        merges = data['model']['merges']
        vocab_size = len(vocab)

        merges_tokens = []
        used_in_merges = set()
        token_to_merge_rank = {}
        for m in merges:
            used_in_merges.update(m.split(' '))
            token = ''.join(m.split(' '))
            if token not in token_to_merge_rank:
                token_to_merge_rank[token] = len(token_to_merge_rank)
            merges_tokens.append(token)

        merges_tokens_set = set(merges_tokens)

        merges_base = [[vocab[token], token] for token in vocab if token not in merges_tokens_set and token in used_in_merges]
        min_rank = min([d[0] for d in merges_base])

        merges_base = [[m[1], m[0] - min_rank] for m in merges_base]
        max_base_rank = max([d[1] for d in merges_base])
        merges_rest = sorted([[d[0], d[1] + max_base_rank + 1] for d in token_to_merge_rank.items()], key=lambda x: x[1])
        tokens_full = merges_base + merges_rest
        tokens_full = [[d[0].replace('▁', ' '), vocab_size - d[1]] for d in tokens_full]

        if args.only_ru:
            print('before ru filter: ', len(tokens_full))
            tokens_full = [d for d in tokens_full if check_ru_token(d[0])]
            print('after ru filter: ', len(tokens_full))

        data = ['\t'.join([d[0], str(d[1])]) for d in tokens_full]
    else:
        raise Exception('incorrect type')
    
    with codecs.open(args.output_path, 'w', 'utf-8') as file:
        file.write('\n'.join(data))
