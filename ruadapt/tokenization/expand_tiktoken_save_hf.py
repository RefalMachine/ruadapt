import tiktoken_ext.openai_public
import tiktoken
from transformers import AutoTokenizer
import os
import json
from .convert_tiktoken import generate_vocab_and_merges
from argparse import ArgumentParser

# Based on Qwen

def custom_tiktoken_extend(tiktoken_base_path, tiktoken_new_path):
    #mergeable_ranks = tiktoken_ext.openai_public.load_tiktoken_bpe('test_tiktoken/tokenizer_extended_test.model')
    mergeable_ranks_base = tiktoken_ext.openai_public.load_tiktoken_bpe(tiktoken_base_path)
    used_ids = set(mergeable_ranks_base.values())
    mergeable_ranks_extend = tiktoken_ext.openai_public.load_tiktoken_bpe(tiktoken_new_path)
    for token, index in mergeable_ranks_extend.items():
        if token in mergeable_ranks_base:
            print(f"extra token {token} exists, skipping")
            continue
        if index in used_ids:
            print(f'the index {index} for extra token {token} exists, skipping')
            continue
        mergeable_ranks_base[token] = index

    special_tokens = {}
    return {
        "name": "custom_tiktoken",
        "pat_str": r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s""",
        "mergeable_ranks": mergeable_ranks_base,
        "special_tokens": special_tokens,
    }

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--tiktoken_base_path')
    parser.add_argument('--tiktoken_new_path')
    parser.add_argument('--output_dir')
    parser.add_argument('--init_output_from', default=None)

    args = parser.parse_args()
        
    tiktoken_base_path = args.tiktoken_base_path
    tiktoken_new_path = args.tiktoken_new_path
    output_dir = args.output_dir

    if args.init_output_from is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.init_output_from)
        tokenizer.save_pretrained(output_dir)

    tiktoken_tokenizer_dict = custom_tiktoken_extend(tiktoken_base_path, tiktoken_new_path)
    tiktoken_tokenizer = tiktoken.core.Encoding(tiktoken_tokenizer_dict.pop('name'), **tiktoken_tokenizer_dict)

    vocab, merges = generate_vocab_and_merges(tiktoken_tokenizer)
    print(len(vocab), len(merges))

    os.remove(os.path.join(output_dir, 'tokenizer.json'))
    with open(os.path.join(output_dir, 'vocab.json'), 'w', encoding='utf-8') as fp:
        json.dump(vocab, fp, ensure_ascii=False)

    with open(os.path.join(output_dir, 'merges.txt'), 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(merges))

    tiktoken_tokenizer = AutoTokenizer.from_pretrained(output_dir)
    tiktoken_tokenizer.save_pretrained(output_dir)