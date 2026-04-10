import argparse
import base64
import collections
import logging
import unicodedata
from pathlib import Path
import regex as re
from tqdm.contrib.logging import tqdm_logging_redirect
import json
import codecs
import heapq

PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG, format="[%(asctime)s] %(levelname)s - %(message)s"
)

def load_tiktoken_bpe(tiktoken_bpe_file: str) -> "dict[bytes, int]":
    contents = open(tiktoken_bpe_file, "rb").read()
    return {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in contents.splitlines() if line)
    }

def dump_tiktoken_bpe(bpe_ranks: "dict[bytes, int]", tiktoken_bpe_file: str) -> None:
    with open(tiktoken_bpe_file, "wb") as f:
        for token, rank in sorted(bpe_ranks.items(), key=lambda x: x[1]):
            token = eval(token)
            f.write(base64.b64encode(token) + b" " + str(rank).encode() + b"\n")

def bytes_to_pieces(the_bytes: bytes) -> "tuple[bytes]":
    return tuple(bytes([byte]) for byte in the_bytes)

def get_pairs(pieces: "tuple[bytes]") -> "set[tuple[bytes, bytes]]":
    return set(zip(pieces[:-1], pieces[1:]))

def apply_bp(pieces: "tuple[bytes, ...]", pair: "tuple[bytes, bytes]") -> "tuple[bytes, ...]":
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

def bpe(word: bytes, merges: "dict[bytes,int]") -> "tuple[bytes, ...]":
    pieces = bytes_to_pieces(word)
    while len(pieces) > 1:
        pairs = get_pairs(pieces)
        pair = min(pairs, key=lambda pair: merges.get(pair[0] + pair[1], float("inf")))
        if pair[0] + pair[1] not in merges:
            break
        pieces = apply_bp(pieces, pair)
    return pieces

def best_pair_sort_key(item: "tuple[tuple[bytes, bytes], int]") -> "tuple[int, int, int, str, bytes]":
    pair, freq = item
    pair_bytes = pair[0] + pair[1]
    pair_byte_length = len(pair_bytes)
    pair_str = pair_bytes.decode("utf-8", errors="replace")
    pair_str_length = len(pair_str)
    return -freq, pair_str_length, pair_byte_length, pair_str, pair_bytes

def learn_bpe_fast(freqs: "dict[str,int]", existing: "dict[bytes, int]") -> "list[tuple[bytes, bytes]]":
    logger.info("Initializing fast BPE structures...")
    vocab = {bpe(k.encode("utf-8"), existing): v for k, v in freqs.items()}
    vocab = {key: value for key, value in vocab.items() if len(key) > 1}
    
    vocab_words = []
    word_freqs = []
    pair_stats = collections.defaultdict(int)
    where = collections.defaultdict(set)
    
    for i, (word_pieces, freq) in enumerate(vocab.items()):
        pieces = list(word_pieces)
        vocab_words.append(pieces)
        word_freqs.append(freq)
        for j in range(len(pieces) - 1):
            pair = (pieces[j], pieces[j+1])
            pair_stats[pair] += freq
            where[pair].add(i)
            
    pq = []
    for pair, freq in pair_stats.items():
        if freq > 0:
            heapq.heappush(pq, (best_pair_sort_key((pair, freq)), pair))
            
    new_merges = []
    logger.info("Starting fast BPE learning loop...")
    
    with tqdm_logging_redirect() as bar:
        while pq:
            sort_key, best_pair = heapq.heappop(pq)
            current_freq = pair_stats.get(best_pair, 0)
            
            if current_freq <= 0:
                continue
                
            expected_key = best_pair_sort_key((best_pair, current_freq))
            if sort_key != expected_key:
                old_freq = -sort_key[0]
                if old_freq > current_freq and current_freq > 0:
                    heapq.heappush(pq, (expected_key, best_pair))
                continue
                
            new_merges.append(best_pair)
            logger.debug(
                f'{best_pair} ({(best_pair[0]+best_pair[1]).decode("utf-8", errors="replace")}) '
                f'is selected as the next merge with freq {current_freq}'
            )
            
            A, B = best_pair
            AB = A + B
            pair_stats[best_pair] = 0
            
            indices = list(where[best_pair])
            del where[best_pair]
            
            words_updated = False
            
            for idx in indices:
                old_pieces = vocab_words[idx]
                freq = word_freqs[idx]
                
                old_counts = collections.defaultdict(int)
                for j in range(len(old_pieces)-1):
                    old_counts[(old_pieces[j], old_pieces[j+1])] += 1
                    
                new_pieces = []
                j = 0
                while j < len(old_pieces):
                    if j < len(old_pieces) - 1 and old_pieces[j] == A and old_pieces[j+1] == B:
                        new_pieces.append(AB)
                        j += 2
                    else:
                        new_pieces.append(old_pieces[j])
                        j += 1
                        
                if len(new_pieces) == len(old_pieces):
                    continue
                    
                vocab_words[idx] = new_pieces
                words_updated = True
                
                new_counts = collections.defaultdict(int)
                for j in range(len(new_pieces)-1):
                    new_counts[(new_pieces[j], new_pieces[j+1])] += 1
                    
                all_pairs = set(old_counts.keys()) | set(new_counts.keys())
                for p in all_pairs:
                    diff = new_counts[p] - old_counts[p]
                    if diff != 0:
                        pair_stats[p] += diff * freq
                        if diff > 0:
                            where[p].add(idx)
                            heapq.heappush(pq, (best_pair_sort_key((p, pair_stats[p])), p))
                        elif pair_stats[p] <= 0:
                            pair_stats[p] = 0
                            
            if words_updated:
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
            except ValueError as _:
                freq = 1
            if word in freqs:
                logger.warning(
                    f"{word} is repeated, the frequency is increased by this much"
                )
                freqs[word] += freq
            else:
                freqs[word] = freq
    return freqs

def make_new_merges_by_bpe(
    input_path: Path, output_path: Path, expand_path: Path, start_id: int
) -> None:
    mergeable_ranks = load_tiktoken_bpe(input_path)
    
    if not start_id or start_id == -1:
        start_id = len(mergeable_ranks)
    elif start_id < len(mergeable_ranks):
        logger.warning(
            f"start_id {start_id} is too small, existing merges will be overridden, DONOT DO THIS. changed to {len(mergeable_ranks)}"
        )
        start_id = len(mergeable_ranks)
        
    expand_vocab_freqs = load_expand_vocab(expand_path)
    for word in list(expand_vocab_freqs):
        token = word.encode("utf-8")
        if token in mergeable_ranks:
            logger.warning(f"word {word} is already a token {token}, skipping")
            del expand_vocab_freqs[word]

    logger.info(f"number of existing merges: {len(mergeable_ranks)}")
    logger.info(f"number of words for expanding: {len(expand_vocab_freqs)}")

    new_merges = learn_bpe_fast(expand_vocab_freqs, mergeable_ranks)
    logger.info(f"number of newly learned merges: {len(new_merges)}")
    
    extra_merges = {str(p[0] + p[1]): str(i) for i, p in enumerate(new_merges, start=start_id)}
    print(len(extra_merges))

    with codecs.open(output_path + '.extra.json', 'w', 'utf-8') as file:
        json.dump(extra_merges, file)

    with codecs.open(output_path + '.extra.json', 'r', 'utf-8') as file:
        data = json.load(file)

    dump_tiktoken_bpe(data, output_path)

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_path", type=str, help="Path for input tiktoken file")
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path for output tiktoken file, containing only the new merges",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        help="Path for words needed adding, each line is a word and its frequency separated by \\t",
    )
    parser.add_argument(
        "--hf_tokenizer_dir",
        type=str,
        default=None,
        help="Path to original HF tokenizer to accurately calculate start_id including special tokens.",
    )
    parser.add_argument(
        "--start_id",
        type=int,
        default=-1,
        help="The start id for new merges. Will be overridden by hf_tokenizer_dir if provided.",
    )

    args = parser.parse_args()

    # Calculate accurate start_id
    if args.hf_tokenizer_dir:
        from transformers import AutoTokenizer
        try:
            tok = AutoTokenizer.from_pretrained(args.hf_tokenizer_dir)
            args.start_id = len(tok)
            logger.info(f"Calculated start_id={args.start_id} from HF tokenizer (includes special tokens)")
        except Exception as e:
            logger.warning(f"Failed to load HF tokenizer for start_id calculation: {e}")

    make_new_merges_by_bpe(
        args.input_path, args.output_path, args.vocab_path, args.start_id
    )

if __name__ == "__main__":
    main()
