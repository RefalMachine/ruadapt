import argparse
import codecs
import numpy as np
import os
import json
import re

def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

byte_encoder = bytes_to_unicode()

def token_bytes_to_string(b):
    return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])

def convert_to_llama(token):
    return token_bytes_to_string(token.replace('▁', ' ').encode())

def check_ru_token(token, min_len):
    # Проверяем, что токен состоит только из кириллицы и пробелов (включая Ё)
    if not bool(re.fullmatch(r'[ А-Яа-яЁё]+', token)):
        return False
    # Считаем длину реального текста (без пробелов, чтобы " по" не прошло фильтр 4)
    clean_token = token.replace(' ', '')
    return len(clean_token) >= min_len

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--type', choices=['unigram', 'bpe', 'from_file'], required=True)
    parser.add_argument('--only_ru', action='store_true')
    parser.add_argument('--min_len', type=int, default=1, help="Минимальная длина буквенной части токена")
    parser.add_argument('--custom_tokens_path', default=None)
    parser.add_argument('--top_k', type=int, default=None, help="Оставить только top_k самых частотных токенов")
    args = parser.parse_args()

    data = []

    if args.type == 'from_file':
        with codecs.open(args.tokenizer_path, 'r', 'utf-8') as file:
            raw_data = file.read().strip().split('\n')
            raw_data = [d.split('\t') for d in raw_data if d]
            data = [[d[0].replace('▁', ' '), d[1]] for d in raw_data]
    else:
        with codecs.open(os.path.join(args.tokenizer_path, 'tokenizer.json'), 'r', 'utf-8') as file:
            tokenizer_json = json.load(file)

        if args.type == 'unigram':
            vocab = tokenizer_json['model']['vocab']
            # В unigram вероятности отрицательные логарифмы
            vocab = [d for d in vocab if d[1] < 0]
            # max(1, ...) защищает от зануления редких токенов
            data = [[d[0].replace('▁', ' '), str(max(1, int(1000000 * np.exp(d[1]))))] for d in vocab]
            
        elif args.type == 'bpe':
            vocab = tokenizer_json['model']['vocab']
            merges = tokenizer_json['model']['merges']
            vocab_size = len(vocab)

            merges_tokens = []
            used_in_merges = set()
            token_to_merge_rank = {}
            for m in merges:
                parts = m.split(' ') if isinstance(m, str) else m
                used_in_merges.update(parts)
                token = ''.join(parts)
                if token not in token_to_merge_rank:
                    token_to_merge_rank[token] = len(token_to_merge_rank)
                merges_tokens.append(token)

            merges_tokens_set = set(merges_tokens)

            merges_base = [[vocab[token], token] for token in vocab if token not in merges_tokens_set and token in used_in_merges]
            if merges_base:
                min_rank = min([d[0] for d in merges_base])
                merges_base = [[m[1], m[0] - min_rank] for m in merges_base]
                max_base_rank = max([d[1] for d in merges_base])
            else:
                max_base_rank = 0

            merges_rest = sorted([[d[0], d[1] + max_base_rank + 1] for d in token_to_merge_rank.items()], key=lambda x: x[1])
            tokens_full = merges_base + merges_rest
            
            # Аппроксимация закона Ципфа вместо линейного убывания (Zipf's law)
            # Чем меньше ранг (d[1]), тем выше частота
            data = [[d[0].replace('▁', ' '), str(max(1, int(10000000 / (d[1] + 1))))] for d in tokens_full]
        
    # Сортируем все токены по "частоте" (второй элемент, парсим как int) по убыванию
    if data:
        data.sort(key=lambda x: int(x[1]), reverse=True)
        if args.top_k is not None:
            data = data[:args.top_k]
            print(f'Retained top {len(data)} tokens based on frequency/score.')

    if args.only_ru:
        print('before ru filter: ', len(data))
        data = [d for d in data if check_ru_token(d[0], args.min_len)]
        print('after ru filter: ', len(data))

    if args.custom_tokens_path is not None:
        with codecs.open(args.custom_tokens_path, 'r', 'utf-8') as file:
            custom_tokens = json.load(file)
            print(f'Loaded {len(custom_tokens)} custom tokens')
            already_added = set([d[0] for d in data])
            custom_tokens = [t for t in custom_tokens if t not in already_added]
            print(f'Added {len(custom_tokens)} custom tokens')
            data += [[t, '1'] for t in custom_tokens]

    data_lines = ['\t'.join(d) for d in data]
    with codecs.open(args.output_path, 'w', 'utf-8') as file:
        file.write('\n'.join(data_lines))
