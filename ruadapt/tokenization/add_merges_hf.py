import argparse
import logging
import unicodedata
from pathlib import Path
import json
import collections
import regex as re
from tqdm.contrib.logging import tqdm_logging_redirect
from tokenizers import Tokenizer

# Pattern used by Qwen to pre-tokenize text
PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG, format="[%(asctime)s] %(levelname)s - %(message)s"
)

def bytes_to_pieces(the_bytes: bytes) -> "tuple[bytes]":
    return tuple(bytes([byte]) for byte in the_bytes)

def get_pairs(pieces: "tuple[bytes]") -> "set[tuple[bytes, bytes]]":
    return set(zip(pieces[:-1], pieces[1:]))

def get_stats(
    vocab: "dict[tuple[bytes, ...], int]",
) -> "dict[tuple[bytes, bytes], int]":
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs

def apply_bp(
    pieces: "tuple[bytes, ...]", pair: "tuple[bytes, bytes]"
) -> "tuple[bytes, ...]":
    new_pieces = []
    first, second = pair
    i = 0
    while i < len(pieces):
        try:
            j = pieces.index(first, i)
            new_pieces.extend(pieces[i:j])
            i = j
        except:
            new_pieces.extend(pieces[i:])
            break

        if pieces[i] == first and i < len(pieces) - 1 and pieces[i + 1] == second:
            new_pieces.append(first + second)
            i += 2
        else:
            new_pieces.append(pieces[i])
            i += 1

    return tuple(new_pieces)

def merge_vocab(
    pair: "tuple[bytes, bytes]", vocab: "dict[tuple[bytes, ...], int]"
) -> "dict[tuple[bytes, ...], int]":
    return {apply_bp(pieces, pair): freq for pieces, freq in vocab.items()}

def bpe(word: bytes, merges: "dict[bytes,int]") -> "tuple[bytes, ...]":
    pieces = bytes_to_pieces(word)
    while len(pieces) > 1:
        pairs = get_pairs(pieces)
        pair = min(pairs, key=lambda pair: merges.get(pair[0] + pair[1], float("inf")))

        if pair[0] + pair[1] not in merges:
            break
        pieces = apply_bp(pieces, pair)
    return pieces

def best_pair_sort_key(
    item: "tuple[dict[bytes, bytes], int]",
) -> "tuple[int, int, int, str, bytes]":
    pair, freq = item
    pair_bytes = pair[0] + pair[1]
    pair_byte_length = len(pair_bytes)
    pair_str = pair_bytes.decode("utf-8", errors="replace")
    pair_str_length = len(pair_str)
    return -freq, pair_str_length, pair_byte_length, pair_str, pair_bytes

def learn_bpe(
    freqs: "dict[str,int]", existing: "dict[bytes, int]"
) -> "tuple[bytes, bytes]":
    vocab = {bpe(k.encode("utf-8"), existing): v for k, v in freqs.items()}
    vocab = {key: value for key, value in vocab.items() if len(key) > 1}
    new_merges = []
    with tqdm_logging_redirect() as bar:
        while vocab:
            pairs = get_stats(vocab)
            if not pairs:
                break

            best, freq = min(pairs.items(), key=best_pair_sort_key)

            logger.debug(
                f'{best} ({(best[0]+best[1]).decode("utf-8", errors="replace")}) is selected as the next merge with freq {freq}'
            )
            new_merges.append(best)

            vocab = merge_vocab(best, vocab)
            vocab = {key: value for key, value in vocab.items() if len(key) > 1}
            bar.update()

    return new_merges

def load_expand_vocab(path: Path) -> "dict[str, int]":
    freqs = {}
    with open(path, "r", encoding="utf8") as fin:
        for line in fin:
            if len(line.strip()) == 0:
                continue
            word, freq = line.split("\t")
            word = unicodedata.normalize("NFC", word)
            parts = re.findall(PAT_STR, word)
            if len(parts) > 1:
                logger.warning(
                    f"{word} would be pre-tokenized to {parts}, and thus cannot be added to vocabulary"
                )
                continue
            try:
                freq = int(freq)
            except ValueError:
                freq = 1
            if word in freqs:
                freqs[word] += freq
            else:
                freqs[word] = freq
    return freqs

def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def make_new_merges_by_bpe(
    hf_tokenizer_dir: str, output_path: str, expand_path: str
) -> None:
    tok = Tokenizer.from_file(str(Path(hf_tokenizer_dir) / "tokenizer.json"))
    model_data = json.load(open(Path(hf_tokenizer_dir) / "tokenizer.json"))["model"]
    str_vocab = model_data["vocab"]
    
    byte_decoder = {v: k for k, v in _bytes_to_unicode().items()}
    
    mergeable_ranks = {}
    for v_str, v_id in str_vocab.items():
        try:
            token_bytes = bytes([byte_decoder[c] for c in v_str])
            mergeable_ranks[token_bytes] = v_id
        except KeyError:
            continue
    
    expand_vocab_freqs = load_expand_vocab(expand_path)
    logger.info(f"number of words for expanding pre: {len(expand_vocab_freqs)}")
    print(list(expand_vocab_freqs.keys())[:100])
    print(list(mergeable_ranks.keys())[:100])
    for word in list(expand_vocab_freqs):
        token = word.encode("utf-8")
        if token in mergeable_ranks:
            logger.warning(f"word {word} is already a token {token}, skipping")
            del expand_vocab_freqs[word]

    logger.info(f"number of existing merges/tokens: {len(mergeable_ranks)}")
    logger.info(f"number of words for expanding: {len(expand_vocab_freqs)}")
    new_merges = learn_bpe(expand_vocab_freqs, mergeable_ranks)
    logger.info(f"number of newly learned merges: {len(new_merges)}")
    
    # Map raw bytes back to unicode strings for HF tokenizer
    byte_encoder = _bytes_to_unicode()
    
    # Update vocab and merges
    max_rank = max(str_vocab.values()) if str_vocab else -1
    new_vocab_items = {}
    new_merges_strs = []
    
    for p1, p2 in new_merges:
        # Convert byte pieces back to HF string tokens
        s1 = "".join([byte_encoder[b] for b in p1])
        s2 = "".join([byte_encoder[b] for b in p2])
        merged_bytes = p1 + p2
        s_merged = "".join([byte_encoder[b] for b in merged_bytes])
        
        merge_str = f"{s1} {s2}"
        if merge_str not in model_data["merges"]:
            model_data["merges"].append(merge_str)
            
        if s_merged not in str_vocab and s_merged not in new_vocab_items:
            max_rank += 1
            new_vocab_items[s_merged] = max_rank
            
    # Add newly merged tokens to vocab
    str_vocab.update(new_vocab_items)
    model_data["vocab"] = str_vocab
    
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the updated tokenizer
    import shutil
    # copy original config files
    for file in Path(hf_tokenizer_dir).glob("*.json"):
        if file.name != "tokenizer.json":
            shutil.copy(file, out_dir / file.name)
            
    tokenizer_json_path = out_dir / "tokenizer.json"
    with open(tokenizer_json_path, "w", encoding="utf-8") as f:
        # Keep original tokenizers structure, just replace the model
        full_data = json.load(open(Path(hf_tokenizer_dir) / "tokenizer.json"))
        full_data["model"] = model_data
        json.dump(full_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved updated tokenizer to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_tokenizer_dir", type=str, required=True, help="Path to base HF tokenizer dir (e.g. Qwen3.5-2B-Base)")
    parser.add_argument("--output_path", type=str, required=True, help="Output dir for new merges")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocab frequencies txt")
    args = parser.parse_args()

    make_new_merges_by_bpe(args.hf_tokenizer_dir, args.output_path, args.vocab_path)
