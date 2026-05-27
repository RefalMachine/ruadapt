import json
from transformers import AutoTokenizer
from tokenizers import Tokenizer
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_tokenizer_path')
    parser.add_argument('--output_path')
    parser.add_argument('--vocab_keep_items', default=0, type=int)
    args = parser.parse_args()

    vocab_keep_items = args.vocab_keep_items
    tokenizer = AutoTokenizer.from_pretrained(args.src_tokenizer_path, use_fast=True)
    assert tokenizer.is_fast, "This only works for fast tokenizers."
    tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
    vocab = tokenizer_json["model"]["vocab"]
    if tokenizer_json["model"]["type"] == "BPE":
        new_vocab = { token: i for token, i in vocab.items() if i < vocab_keep_items }
        merges = tokenizer_json["model"]["merges"]
        new_merges = []
        for i in range(len(merges)):
            a, b = merges[i].split()
            new_token = "".join((a, b))
            if a in new_vocab and b in new_vocab and new_token in new_vocab:
                new_merges.append(merges[i])
        tokenizer_json["model"]["merges"] = new_merges
    elif tokenizer_json["model"]["type"] == "Unigram":
        new_vocab = vocab[:vocab_keep_items]
    elif tokenizer_json["model"]["type"] == "WordPiece" or tokenizer_json["model"]["type"] == "WordLevel":
        new_vocab = { token: i for token, i in vocab.items() if i < vocab_keep_items }
    else:
        raise ValueError(f"don't know how to handle {tokenizer_json['model']['type']}")
    tokenizer_json["model"]["vocab"] = new_vocab
    tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))
    tokenizer.save_pretrained(args.output_path)