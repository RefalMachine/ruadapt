import tiktoken_ext.openai_public
import tiktoken
from transformers import AutoTokenizer
import os
import json
from .convert_tiktoken import generate_vocab_and_merges
from argparse import ArgumentParser
import re
import base64
# Based on Qwen
def load_tiktoken_bpe(tiktoken_bpe_file):
    # NB: do not add caching to this function
    with open(tiktoken_bpe_file, 'rb') as file:
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

def custom_tiktoken_extend(tiktoken_base_path, tiktoken_new_path):
    #mergeable_ranks = tiktoken_ext.openai_public.load_tiktoken_bpe('test_tiktoken/tokenizer_extended_test.model')
    mergeable_ranks_base = load_tiktoken_bpe(tiktoken_base_path)
    used_ids = set(mergeable_ranks_base.values())
    mergeable_ranks_extend = load_tiktoken_bpe(tiktoken_new_path)
    print(len(mergeable_ranks_base))
    print(len(mergeable_ranks_extend))
    print(tiktoken_new_path)
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

def check_contains_digit(t):
    m = re.match('[0-9]+', t)
    return m is not None

def check_if_number(t):
    m = re.match('[0-9]+', t)
    if m is None:
        return False
    return len(m[0]) == len(t) and len(t) > 1

def filter_numbers(vocab, merges):
    merges = [m for m in merges if not check_contains_digit(m)]
    vocab = sorted([[v, i] for v, i in vocab.items()], key=lambda x: x[1])
    vocab = [v[0] for v in vocab if not check_if_number(v[0])]
    vocab = {t: i for i, t in enumerate(vocab)}
    return vocab, merges


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--tiktoken_base_path')
    parser.add_argument('--tiktoken_new_path')
    parser.add_argument('--output_dir')
    parser.add_argument('--init_output_from', default=None)
    parser.add_argument('--filter_numbers', action='store_true')

    args = parser.parse_args()
        
    tiktoken_base_path = args.tiktoken_base_path
    tiktoken_new_path = args.tiktoken_new_path
    output_dir = args.output_dir

    if args.init_output_from is not None:
        import shutil
        os.makedirs(output_dir, exist_ok=True)
        # Copy original HF files to preserve chat_template and proper tokenizer_config.json
        # Only copy what's absolutely necessary. DO NOT copy vocab.json and merges.txt 
        # because AutoTokenizer might prioritize them over tokenizer.json!
        for filename in ['tokenizer_config.json', 'tokenizer.json', 'special_tokens_map.json']:
            src = os.path.join(args.init_output_from, filename)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(output_dir, filename))

    tiktoken_tokenizer_dict = custom_tiktoken_extend(tiktoken_base_path, tiktoken_new_path)
    tiktoken_tokenizer = tiktoken.core.Encoding(tiktoken_tokenizer_dict.pop('name'), **tiktoken_tokenizer_dict)

    vocab, merges = generate_vocab_and_merges(tiktoken_tokenizer)
    if args.filter_numbers:
        vocab, merges = filter_numbers(vocab, merges) 

    # We must patch tokenizer.json carefully to preserve special tokens and original merges
    tokenizer_json_path = os.path.join(output_dir, 'tokenizer.json')
    if os.path.exists(tokenizer_json_path):
        with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
            
        original_vocab = full_data["model"]["vocab"]
        original_merges = full_data["model"]["merges"]
        
        # Calculate max original ID (base vocab + special added tokens)
        max_orig_id = max(original_vocab.values())
        if "added_tokens" in full_data:
            max_orig_id = max(max_orig_id, max(t["id"] for t in full_data["added_tokens"]))
            
        # Append only NEW vocab items (ID > max_orig_id)
        new_vocab_count = 0
        for k, v in vocab.items():
            if v > max_orig_id:
                original_vocab[k] = v
                new_vocab_count += 1
                
        # HF save_pretrained() sometimes serializes merges as lists of strings instead of space-separated strings.
        # We normalize them back to standard space-separated strings.
        normalized_merges = []
        for m in original_merges:
            if isinstance(m, list):
                normalized_merges.append(" ".join(m))
            else:
                normalized_merges.append(str(m))
        original_merges = normalized_merges
            
        # Append only NEW merges that are STRICTLY CYRILLIC
        orig_merges_set = set(original_merges)
        new_merges_count = 0
        
        # Helper to decode GPT2 format to text to check for cyrillic
        def is_cyrillic_merge(merge_str):
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
            b2c = dict(zip(bs, cs))
            c2b = {v: k for k, v in b2c.items()}
            
            p1, p2 = merge_str.split(" ", 1)
            b1 = bytes([c2b.get(c, 0) for c in p1])
            b2 = bytes([c2b.get(c, 0) for c in p2])
            b_full = b1 + b2
            
            try:
                text = b_full.decode("utf-8", errors="strict")
                import re
                # MUST contain at least one Cyrillic character.
                # If it's just English letters, numbers, or punctuation -> reject.
                if not re.search(r"[А-Яа-яЁё]", text):
                    return False
                return True
            except UnicodeDecodeError:
                return False

        for m in merges:
            if m not in orig_merges_set:
                if is_cyrillic_merge(m):
                    original_merges.append(m)
                    new_merges_count += 1
                
        print(f"Injected {new_vocab_count} new tokens and {new_merges_count} new strictly-safe merges into HF tokenizer.")
        
        full_data["model"]["vocab"] = original_vocab
        full_data["model"]["merges"] = original_merges
        
        with open(tokenizer_json_path, 'w', encoding='utf-8') as f:
            json.dump(full_data, f, ensure_ascii=False, indent=2)
            
        # VERY IMPORTANT: Delete vocab.json and merges.txt if they exist.
        # If we leave them here, HuggingFace AutoTokenizer falls back to the slow, buggy Python
        # Qwen2Tokenizer which breaks Hindi/Arabic unicode BPE merges. 
        # By removing them, we force HF to load tokenizer.json via the fast Rust TokenizersBackend.
        for legacy_file in ['vocab.json', 'merges.txt']:
            lf_path = os.path.join(output_dir, legacy_file)
            if os.path.exists(lf_path):
                os.remove(lf_path)
    else:
        print("WARNING: tokenizer.json not found, falling back to raw txt output.")
        with open(os.path.join(output_dir, 'vocab.json'), 'w', encoding='utf-8') as fp:
            json.dump(vocab, fp, ensure_ascii=False)
        with open(os.path.join(output_dir, 'merges.txt'), 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(merges))