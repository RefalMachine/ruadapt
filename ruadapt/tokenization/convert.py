"""Format converters: tiktoken <-> HF, vocab extraction, frequency lists."""

import argparse
import base64
import codecs
import json
import os
import re
from typing import Dict, List, Optional

import numpy as np
import tiktoken
from transformers import AutoTokenizer


def bytes_to_unicode() -> Dict[int, str]:
    """Returns the GPT-2 byte-to-unicode mapping."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
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


_byte_encoder = bytes_to_unicode()


def token_bytes_to_string(b: bytes) -> str:
    """Convert token bytes to GPT-2 unicode string representation."""
    return "".join([_byte_encoder[ord(char)] for char in b.decode("latin-1")])


def convert_to_llama(token: str) -> str:
    """Convert a SentencePiece-style token to GPT-2 byte representation."""
    return token_bytes_to_string(token.replace("▁", " ").encode())


def check_ru_token(token: str, min_len: int = 1) -> bool:
    """Check if token consists only of Cyrillic characters and spaces."""
    if not bool(re.fullmatch(r"[ А-Яа-яЁё]+", token)):
        return False
    clean_token = token.replace(" ", "")
    return len(clean_token) >= min_len


def check_contains_digit(t: str) -> bool:
    return re.search(r"[0-9]", t) is not None


def check_if_number(t: str) -> bool:
    m = re.match(r"[0-9]+", t)
    if m is None:
        return False
    return len(m[0]) == len(t) and len(t) > 1


def filter_numbers(vocab: Dict[str, int], merges: List[str]):
    """Remove numeric tokens and merge rules containing digits."""
    merges = [m for m in merges if not check_contains_digit(m)]
    vocab = sorted([[v, i] for v, i in vocab.items()], key=lambda x: x[1])
    vocab = [v[0] for v in vocab if not check_if_number(v[0])]
    vocab = {t: i for i, t in enumerate(vocab)}
    return vocab, merges


# --- tiktoken conversion ---


def bpe(mergeable_ranks: Dict[bytes, int], token: bytes, max_rank: Optional[int] = None) -> List[bytes]:
    """Run BPE on a single token using tiktoken mergeable ranks."""
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts


def generate_vocab_and_merges(encoder) -> tuple:
    """Generate HF-compatible vocab and merges from a tiktoken encoder."""
    mergeable_ranks = encoder._mergeable_ranks

    merges = []
    vocab = {}
    for token, rank in mergeable_ranks.items():
        vocab[token_bytes_to_string(token)] = rank

        if len(token) == 1:
            continue
        merged = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        if len(merged) != 2:
            continue

        merges.append(" ".join(map(token_bytes_to_string, merged)))

    vocab.update(encoder._special_tokens)

    return vocab, merges


def load_tiktoken_bpe(tiktoken_bpe_file: str) -> Dict[bytes, int]:
    """Load a tiktoken BPE file (base64-encoded token per line)."""
    with open(tiktoken_bpe_file, "rb") as file:
        contents = file.read()
    ret = {}
    for line in contents.splitlines():
        if not line:
            continue
        try:
            token, rank = line.split()
            ret[base64.b64decode(token)] = int(rank)
        except Exception as e:
            raise ValueError(f"Error parsing line {line!r} in {tiktoken_bpe_file}") from e
    return ret


def custom_tiktoken_extend(tiktoken_base_path: str, tiktoken_new_path: str) -> dict:
    """Extend a tiktoken tokenizer with additional tokens from another file."""
    mergeable_ranks_base = load_tiktoken_bpe(tiktoken_base_path)
    used_ids = set(mergeable_ranks_base.values())
    mergeable_ranks_extend = load_tiktoken_bpe(tiktoken_new_path)

    for token, index in mergeable_ranks_extend.items():
        if token in mergeable_ranks_base:
            continue
        if index in used_ids:
            continue
        mergeable_ranks_base[token] = index

    special_tokens = {}
    return {
        "name": "custom_tiktoken",
        "pat_str": r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s""",
        "mergeable_ranks": mergeable_ranks_base,
        "special_tokens": special_tokens,
    }


# --- CLI entrypoints ---


def cli_extract_vocab():
    """Extract vocab from a tokenizer to a frequency list file."""
    parser = argparse.ArgumentParser(description="Extract vocab from tokenizer to frequency list")
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--type", choices=["unigram", "bpe", "from_file"], required=True)
    parser.add_argument("--only_ru", action="store_true")
    parser.add_argument("--min_len", type=int, default=1, help="Min length of Cyrillic part of token")
    parser.add_argument("--custom_tokens_path", default=None)
    parser.add_argument("--top_k", type=int, default=None, help="Keep only top_k most frequent tokens")
    args = parser.parse_args()

    data = []

    if args.type == "from_file":
        with codecs.open(args.tokenizer_path, "r", "utf-8") as file:
            raw_data = file.read().strip().split("\n")
            raw_data = [d.split("\t") for d in raw_data if d]
            data = [[d[0].replace("▁", " "), d[1]] for d in raw_data]
    else:
        with codecs.open(os.path.join(args.tokenizer_path, "tokenizer.json"), "r", "utf-8") as file:
            tokenizer_json = json.load(file)

        if args.type == "unigram":
            vocab = tokenizer_json["model"]["vocab"]
            vocab = [d for d in vocab if d[1] < 0]
            data = [[d[0].replace("▁", " "), str(max(1, int(1000000 * np.exp(d[1]))))] for d in vocab]

        elif args.type == "bpe":
            vocab = tokenizer_json["model"]["vocab"]
            merges = tokenizer_json["model"]["merges"]

            merges_tokens = []
            used_in_merges = set()
            token_to_merge_rank = {}
            for m in merges:
                parts = m.split(" ") if isinstance(m, str) else m
                used_in_merges.update(parts)
                token = "".join(parts)
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

            data = [[d[0].replace("▁", " "), str(max(1, int(10000000 / (d[1] + 1))))] for d in tokens_full]

    if data:
        data.sort(key=lambda x: int(x[1]), reverse=True)
        if args.top_k is not None:
            data = data[: args.top_k]
            print(f"Retained top {len(data)} tokens based on frequency/score.")

    if args.only_ru:
        print("before ru filter: ", len(data))
        data = [d for d in data if check_ru_token(d[0], args.min_len)]
        print("after ru filter: ", len(data))

    if args.custom_tokens_path is not None:
        with codecs.open(args.custom_tokens_path, "r", "utf-8") as file:
            custom_tokens = json.load(file)
            print(f"Loaded {len(custom_tokens)} custom tokens")
            already_added = set([d[0] for d in data])
            custom_tokens = [t for t in custom_tokens if t not in already_added]
            print(f"Added {len(custom_tokens)} custom tokens")
            data += [[t, "1"] for t in custom_tokens]

    data_lines = ["\t".join(d) for d in data]
    with codecs.open(args.output_path, "w", "utf-8") as file:
        file.write("\n".join(data_lines))


def cli_expand_tiktoken():
    """Extend a tiktoken tokenizer and save in HF format."""
    parser = argparse.ArgumentParser(description="Extend tiktoken tokenizer and save as HF")
    parser.add_argument("--tiktoken_base_path", required=True)
    parser.add_argument("--tiktoken_new_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--init_output_from", default=None)
    parser.add_argument("--filter_numbers", action="store_true")
    args = parser.parse_args()

    if args.init_output_from is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.init_output_from)
        tokenizer.save_pretrained(args.output_dir)

    tiktoken_tokenizer_dict = custom_tiktoken_extend(args.tiktoken_base_path, args.tiktoken_new_path)
    tiktoken_tokenizer = tiktoken.core.Encoding(tiktoken_tokenizer_dict.pop("name"), **tiktoken_tokenizer_dict)

    vocab, merges = generate_vocab_and_merges(tiktoken_tokenizer)
    if args.filter_numbers:
        vocab, merges = filter_numbers(vocab, merges)

    print(len(vocab), len(merges))

    os.remove(os.path.join(args.output_dir, "tokenizer.json"))
    with open(os.path.join(args.output_dir, "vocab.json"), "w", encoding="utf-8") as fp:
        json.dump(vocab, fp, ensure_ascii=False)

    with open(os.path.join(args.output_dir, "merges.txt"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(merges))

    tiktoken_tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    tiktoken_tokenizer.save_pretrained(args.output_dir)


def cli_convert_hf_vocab_to_freq_list():
    """Convert HF tokenizer vocab to a frequency list file."""
    parser = argparse.ArgumentParser(description="Convert HF tokenizer vocab to frequency list")
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--type", required=True)
    parser.add_argument("--only_ru", action="store_true")
    parser.add_argument("--custom_tokens_path", default=None)
    args = parser.parse_args()

    if args.type == "from_file":
        with codecs.open(args.tokenizer_path, "r", "utf-8") as file:
            data = file.read().strip().split("\n")
        data = [d.split("\t") for d in data]
        data = [[d[0].replace("▁", " "), d[1]] for d in data]
    else:
        with codecs.open(os.path.join(args.tokenizer_path, "tokenizer.json"), "r", "utf-8") as file:
            data = json.load(file)

    if args.type == "unigram":
        data = data["model"]["vocab"]
        data = [d for d in data if d[1] < 0]
        data = [[d[0].replace("▁", " "), str(int(1000000 * np.exp(d[1])))] for d in data]

    elif args.type == "bpe":
        print("WARNING: frequency calculation may be incorrect!")

        vocab = data["model"]["vocab"]
        merges = data["model"]["merges"]
        vocab_size = len(vocab)

        merges_tokens = []
        used_in_merges = set()
        token_to_merge_rank = {}
        for m in merges:
            used_in_merges.update(m.split(" "))
            token = "".join(m.split(" "))
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
        tokens_full = [[d[0].replace("▁", " "), vocab_size - d[1]] for d in tokens_full]

        data = [[d[0], str(d[1])] for d in tokens_full]

    elif args.type == "from_file":
        pass
    else:
        raise ValueError("incorrect type")

    if args.only_ru:
        print("before ru filter: ", len(data))
        data = [d for d in data if check_ru_token(d[0])]
        print("after ru filter: ", len(data))

    if args.custom_tokens_path is not None:
        with codecs.open(args.custom_tokens_path, "r", "utf-8") as file:
            custom_tokens = json.load(file)
        print(f"Loaded {len(custom_tokens)} custom tokens")
        already_added = set([d[0] for d in data])
        custom_tokens = [t for t in custom_tokens if t not in already_added]
        print(f"Added {len(custom_tokens)} custom tokens")
        data += [[t, "1"] for t in custom_tokens]

    data = ["\t".join(d) for d in data]
    with codecs.open(args.output_path, "w", "utf-8") as file:
        file.write("\n".join(data))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m ruadapt.tokenization.convert <subcommand> [args]")
        print("Subcommands: cli_extract_vocab, cli_expand_tiktoken, cli_convert_hf_vocab_to_freq_list")
        sys.exit(1)

    subcommand = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    dispatch = {
        "cli_extract_vocab": cli_extract_vocab,
        "cli_expand_tiktoken": cli_expand_tiktoken,
        "cli_convert_hf_vocab_to_freq_list": cli_convert_hf_vocab_to_freq_list,
    }

    if subcommand not in dispatch:
        print(f"Unknown subcommand: {subcommand}")
        print(f"Available: {', '.join(dispatch.keys())}")
        sys.exit(1)

    dispatch[subcommand]()
