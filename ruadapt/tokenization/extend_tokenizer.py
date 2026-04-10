import argparse
import collections
import heapq
import json
import logging
import os
import shutil
import unicodedata
from typing import Dict, List, Tuple, Set

import regex as re_lib

logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PAT_STR = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def gpt2_bytes_to_unicode() -> Dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return {k: chr(v) for k, v in zip(bs, cs)}


def bytes_to_pieces(the_bytes: bytes) -> Tuple[bytes, ...]:
    return tuple(bytes([byte]) for byte in the_bytes)


def get_pairs(pieces: Tuple[bytes, ...]) -> Set[Tuple[bytes, bytes]]:
    return set(zip(pieces[:-1], pieces[1:]))


def apply_bp(pieces: Tuple[bytes, ...], pair: Tuple[bytes, bytes]) -> Tuple[bytes, ...]:
    new_pieces = []
    first, second = pair
    i = 0
    while i < len(pieces):
        try:
            j = pieces.index(first, i)
            new_pieces.extend(pieces[i:j])
            i = j
        except ValueError:
            new_pieces.extend(pieces[i:])
            break
        if pieces[i] == first and i < len(pieces) - 1 and pieces[i + 1] == second:
            new_pieces.append(first + second)
            i += 2
        else:
            new_pieces.append(pieces[i])
            i += 1
    return tuple(new_pieces)


def bpe(word: bytes, merges: Dict[bytes, int]) -> Tuple[bytes, ...]:
    pieces = bytes_to_pieces(word)
    while len(pieces) > 1:
        pairs = get_pairs(pieces)
        pair = min(pairs, key=lambda p: merges.get(p[0] + p[1], float("inf")))
        if pair[0] + pair[1] not in merges:
            break
        pieces = apply_bp(pieces, pair)
    return pieces


def best_pair_sort_key(item: Tuple[Tuple[bytes, bytes], int]) -> Tuple[int, int, int, str, bytes]:
    pair, freq = item
    pair_bytes = pair[0] + pair[1]
    pair_byte_length = len(pair_bytes)
    pair_str = pair_bytes.decode("utf-8", errors="replace")
    pair_str_length = len(pair_str)
    return -freq, pair_str_length, pair_byte_length, pair_str, pair_bytes


def learn_bpe_fast(freqs: Dict[str, int], existing: Dict[bytes, int]) -> List[Tuple[bytes, bytes]]:
    logger.info("Initializing fast BPE structures...")

    collapsed_count = 0
    vocab: Dict[Tuple[bytes, ...], int] = {}
    for word_str, freq in freqs.items():
        pieces = bpe(word_str.encode("utf-8"), existing)
        if len(pieces) == 1:
            collapsed_count += 1
            logger.debug(f"Word collapsed to existing token, skipping: {word_str!r} -> {pieces[0]!r}")
        else:
            vocab[pieces] = vocab.get(pieces, 0) + freq
    if collapsed_count:
        logger.warning(f"{collapsed_count} words collapsed to single token, excluded from BPE training.")

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

        A, B = best_pair
        AB = A + B
        pair_stats[best_pair] = 0

        indices = list(where[best_pair])
        del where[best_pair]

        for idx in indices:
            old_pieces = vocab_words[idx]
            freq = word_freqs[idx]

            old_counts = collections.defaultdict(int)
            for j in range(len(old_pieces) - 1):
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

            new_counts = collections.defaultdict(int)
            for j in range(len(new_pieces) - 1):
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
                        where.pop(p, None)

    return new_merges


def load_expand_vocab(path: str) -> Dict[str, int]:
    freqs: Dict[str, int] = {}
    skipped_multi_part = 0
    pat = re_lib.compile(PAT_STR)

    with open(path, "r", encoding="utf8") as fin:
        for line in fin:
            if not line.strip():
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            word, freq_str = parts
            word = unicodedata.normalize("NFC", word)

            pre_parts = re_lib.findall(pat, word)
            if len(pre_parts) > 1:
                skipped_multi_part += 1
                continue

            try:
                freq = int(freq_str)
            except (ValueError, IndexError):
                freq = 1

            freqs[word] = freqs.get(word, 0) + freq

    if skipped_multi_part:
        logger.info(f"Skipped {skipped_multi_part} multi-part words (cross GPT-2 pre-tok boundary).")

    return freqs


def load_merges_from_json(tokenizer_json_path: str, str_to_byte_map: Dict[str, bytes]) -> Dict[bytes, int]:
    merge_ranks: Dict[bytes, int] = {}

    for b in range(256):
        merge_ranks[bytes([b])] = b

    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    merges_list = data.get("model", {}).get("merges", [])

    rank = 256
    skipped = 0
    for merge_str in merges_list:
        parts = merge_str.split(" ")
        if len(parts) == 2:
            try:
                p1 = b"".join(str_to_byte_map[c] for c in parts[0])
                p2 = b"".join(str_to_byte_map[c] for c in parts[1])
                merge_ranks[p1 + p2] = rank
                rank += 1
            except KeyError as e:
                skipped += 1
                logger.warning(f"Unknown GPT-2 char {e} in merge rule {merge_str!r}, skipping.")

    if skipped:
        logger.warning(f"Skipped {skipped} merge rules due to unknown GPT-2 characters.")

    return merge_ranks


def _patch_id_in_obj(obj, old_to_new: Dict[int, int], path: str = ""):
    """Recursively walk a JSON-serializable object and remap integer IDs."""
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if isinstance(v, int) and v in old_to_new:
                logger.info(f"  {path}.{k}: {v} -> {old_to_new[v]}")
                obj[k] = old_to_new[v]
            else:
                _patch_id_in_obj(v, old_to_new, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, int) and v in old_to_new:
                logger.info(f"  {path}[{i}]: {v} -> {old_to_new[v]}")
                obj[i] = old_to_new[v]
            else:
                _patch_id_in_obj(v, old_to_new, f"{path}[{i}]")


def main():
    parser = argparse.ArgumentParser(
        description="Extend tokenizer: inject Russian tokens BEFORE added tokens (defragmented layout)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base_tokenizer", type=str, required=True)
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    b2c = gpt2_bytes_to_unicode()
    c2b = {v: bytes([k]) for k, v in b2c.items()}

    tokenizer_json_path = os.path.join(args.base_tokenizer, "tokenizer.json")
    if not os.path.exists(tokenizer_json_path):
        raise FileNotFoundError(f"tokenizer.json not found in {args.base_tokenizer}")

    # ── 1. Learn new BPE merges ──────────────────────────────────────────
    logger.info("Loading byte-level merge ranks from tokenizer.json...")
    merge_ranks = load_merges_from_json(tokenizer_json_path, c2b)
    logger.info(f"Loaded {len(merge_ranks)} merge ranks/vocab tokens.")

    logger.info("Loading expansion vocabulary...")
    expand_vocab = load_expand_vocab(args.vocab)

    already_present = 0
    for word in list(expand_vocab):
        if word.encode("utf-8") in merge_ranks:
            del expand_vocab[word]
            already_present += 1
    if already_present:
        logger.info(f"Skipped {already_present} words already in base tokenizer.")

    logger.info(f"Expansion vocabulary: {len(expand_vocab)} words.")
    new_merges = learn_bpe_fast(expand_vocab, merge_ranks)
    logger.info(f"Learned {len(new_merges)} new merge rules.")

    # ── 2. Load base files ───────────────────────────────────────────────
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vocab: Dict[str, int] = dict(data["model"]["vocab"])
    merges: List[str] = list(data["model"]["merges"])

    # Original added_tokens from tokenizer.json (only these go into output tokenizer.json)
    added_tokens_tj: List[dict] = sorted(data.get("added_tokens", []), key=lambda t: t["id"])
    added_tokens_tj_ids = {t["id"] for t in added_tokens_tj}

    # Load tokenizer_config.json to find ghost tokens (decoder-only)
    tok_cfg_src = os.path.join(args.base_tokenizer, "tokenizer_config.json")
    tok_cfg = None
    ghost_tokens: List[dict] = []
    if os.path.exists(tok_cfg_src):
        with open(tok_cfg_src, "r") as f:
            tok_cfg = json.load(f)

        if "added_tokens_decoder" in tok_cfg:
            for id_str, entry_data in tok_cfg["added_tokens_decoder"].items():
                tid = int(id_str)
                if tid not in added_tokens_tj_ids:
                    ghost_tokens.append({
                        "id": tid,
                        "content": entry_data.get("content", ""),
                        "special": entry_data.get("special", False),
                    })
                    logger.debug(
                        f"Ghost token (decoder-only): ID={tid} "
                        f"content={entry_data.get('content','')!r}"
                    )

    base_vocab_max = max(vocab.values())
    num_tj = len(added_tokens_tj)
    num_ghost = len(ghost_tokens)
    num_all = num_tj + num_ghost

    logger.info(f"Base vocab: {len(vocab)} entries, max ID = {base_vocab_max}")
    logger.info(f"Tokens to shift: {num_all} total")
    logger.info(f"  - tokenizer.json added_tokens: {num_tj}")
    logger.info(f"  - decoder ghost tokens: {num_ghost}")

    # ── 3. Inject Russian tokens into vocab ──────────────────────────────
    next_id = base_vocab_max + 1
    merges_set = set(merges)
    new_vocab_count = 0
    new_merges_count = 0

    for pair in new_merges:
        left_str = "".join(b2c[b] for b in pair[0])
        right_str = "".join(b2c[b] for b in pair[1])
        merged_bytes = pair[0] + pair[1]
        merged_str = "".join(b2c[b] for b in merged_bytes)

        rule = f"{left_str} {right_str}"
        if rule not in merges_set:
            merges.append(rule)
            merges_set.add(rule)
            new_merges_count += 1

        if merged_str not in vocab:
            vocab[merged_str] = next_id
            next_id += 1
            new_vocab_count += 1

    new_tokens_end = next_id - 1
    logger.info(f"Assigned {new_vocab_count} new tokens to vocab (IDs {base_vocab_max + 1}..{new_tokens_end})")

    # ── 4. Build old->new ID map and shift ALL added tokens ──────────────
    # We shift BOTH tokenizer.json tokens AND ghost tokens to avoid collisions.
    old_to_new: Dict[int, int] = {}

    # Shift tokenizer.json added_tokens (these go into output tokenizer.json)
    for i, entry in enumerate(added_tokens_tj):
        old_id = entry["id"]
        new_id = next_id + i
        old_to_new[old_id] = new_id
        entry["id"] = new_id
        # Remove from vocab if present (maintain vocab/additional separation)
        if entry["content"] in vocab:
            logger.warning(f"Added token {entry['content']!r} was in vocab — removing.")
            del vocab[entry["content"]]

    # Shift ghost tokens (these stay ONLY in tokenizer_config.json, not in tokenizer.json)
    ghost_start = next_id + num_tj
    for i, entry in enumerate(ghost_tokens):
        old_id = entry["id"]
        new_id = ghost_start + i
        old_to_new[old_id] = new_id

    final_id = next_id + num_all - 1
    logger.info(f"Shifted IDs: {next_id}..{final_id}")
    logger.info(f"  tokenizer.json tokens: {next_id}..{next_id + num_tj - 1}")
    logger.info(f"  ghost tokens:          {ghost_start}..{final_id}")

    # ── 5. Save tokenizer.json (ONLY original tokens, NO ghosts) ─────────
    data["model"]["vocab"] = vocab
    data["model"]["merges"] = merges
    # Only original tokenizer.json tokens — ghosts stay out
    data["added_tokens"] = added_tokens_tj

    out_path = os.path.join(args.output_dir, "tokenizer.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logger.info(
        f"Saved tokenizer.json: {len(vocab)} vocab entries, "
        f"{len(merges)} merges, {len(added_tokens_tj)} added_tokens (no ghosts)"
    )

    # ── 6. Patch config.json ─────────────────────────────────────────────
    # vocab_size must cover the maximum ID across vocab + added_tokens + ghosts
    max_id_all = max(max(vocab.values()), final_id)
    raw_vocab_size = max_id_all + 1
    padded_vocab_size = ((raw_vocab_size + 255) // 256) * 256
    logger.info(f"vocab_size: {raw_vocab_size} (raw) -> {padded_vocab_size} (padded to 256)")

    cfg_src = os.path.join(args.base_tokenizer, "config.json")
    cfg_dst = os.path.join(args.output_dir, "config.json")
    if os.path.exists(cfg_src):
        with open(cfg_src, "r") as f:
            cfg = json.load(f)

        cfg["vocab_size"] = padded_vocab_size
        if "text_config" in cfg and isinstance(cfg["text_config"], dict):
            cfg["text_config"]["vocab_size"] = padded_vocab_size

        _patch_id_in_obj(cfg, old_to_new, "config")

        with open(cfg_dst, "w") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved patched config.json")
    else:
        logger.warning("config.json not found — skipping.")

    # ── 7. Patch tokenizer_config.json (remap decoder IDs) ───────────────
    tok_cfg_dst = os.path.join(args.output_dir, "tokenizer_config.json")
    if tok_cfg is not None:
        if "added_tokens_decoder" in tok_cfg:
            new_decoder = {}
            for old_id_str, entry in tok_cfg["added_tokens_decoder"].items():
                old_id = int(old_id_str)
                if old_id in old_to_new:
                    new_decoder[str(old_to_new[old_id])] = entry
                else:
                    new_decoder[old_id_str] = entry
            tok_cfg["added_tokens_decoder"] = new_decoder

        # Force TokenizersBackend (critical for Hindi/Arabic)
        tok_cfg["tokenizer_class"] = "TokenizersBackend"

        with open(tok_cfg_dst, "w") as f:
            json.dump(tok_cfg, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved patched tokenizer_config.json")
    else:
        logger.warning("tokenizer_config.json not found — skipping.")

    # ── 8. Copy special_tokens_map.json ──────────────────────────────────
    for fname in ["special_tokens_map.json"]:
        src = os.path.join(args.base_tokenizer, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output_dir, fname))
            logger.info(f"Copied {fname}.")

    # ── 9. Summary ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("FINAL LAYOUT:")
    logger.info(f"  vocab (BPE):           0 .. {base_vocab_max}")
    logger.info(f"  new Russian tokens:    {base_vocab_max + 1} .. {new_tokens_end}  (in vocab)")
    logger.info(f"  added_tokens (tj):     {next_id} .. {next_id + num_tj - 1}  (in tokenizer.json added_tokens)")
    logger.info(f"  ghost tokens:          {ghost_start} .. {final_id}  (in tokenizer_config.json decoder only)")
    logger.info(f"  vocab_size (padded):   {padded_vocab_size}")
    logger.info("=" * 60)
    logger.info(f"Done. Defragmented tokenizer saved to {args.output_dir}")


if __name__ == "__main__":
    main()
