import argparse
import os
import json
import time
import multiprocessing as mp
from tqdm import tqdm
from transformers import AutoTokenizer

from ruadapt.tokenization.utils import get_first_diff_id, get_special_token_ids, get_new_token_ids

# --- WORKER LOGIC ---
WORKER_TOKENIZER = None
WORKER_FIRST_DIFF_ID = None
WORKER_SPECIAL_IDS = None
WORKER_MAX_CHARS = None
WORKER_MAX_TOKENS = None
WORKER_TRUNCATE = None

def init_worker(tokenizer_path, first_diff_id, special_ids, max_chars, max_tokens, truncate_tokens):
    global WORKER_TOKENIZER, WORKER_FIRST_DIFF_ID, WORKER_SPECIAL_IDS
    global WORKER_MAX_CHARS, WORKER_MAX_TOKENS, WORKER_TRUNCATE
    
    WORKER_TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path)
    WORKER_FIRST_DIFF_ID = first_diff_id
    WORKER_SPECIAL_IDS = special_ids
    WORKER_MAX_CHARS = max_chars
    WORKER_MAX_TOKENS = max_tokens
    WORKER_TRUNCATE = truncate_tokens

def process_doc_worker(doc_json_str):
    try:
        doc = json.loads(doc_json_str)
    except:
        return None
        
    text = doc.get("text", "")
    domain = doc.get("domain", "")
    
    is_en = domain == "enwiki" or domain == "en" or domain.startswith("en_")
    if is_en or len(text) > WORKER_MAX_CHARS:
        return None
        
    input_ids = WORKER_TOKENIZER.encode(text, add_special_tokens=False)
    scanned_tokens = len(input_ids)
    
    if len(input_ids) > WORKER_MAX_TOKENS:
        return {"valid": False, "scanned_tokens": scanned_tokens, "saved_tokens": 0}
        
    if WORKER_TRUNCATE > 0:
        input_ids = input_ids[:WORKER_TRUNCATE]
        
    saved_tokens = len(input_ids)
        
    counts = {}
    for t_id in input_ids:
        if t_id >= WORKER_FIRST_DIFF_ID and t_id not in WORKER_SPECIAL_IDS:
            counts[t_id] = counts.get(t_id, 0) + 1
            
    if not counts:
        return {"valid": False, "scanned_tokens": scanned_tokens, "saved_tokens": 0}
        
    if WORKER_TRUNCATE > 0 and len(input_ids) >= WORKER_TRUNCATE:
        text = WORKER_TOKENIZER.decode(input_ids)
        
    return {
        "valid": True, 
        "text": text, 
        "counts": counts, 
        "scanned_tokens": scanned_tokens, 
        "saved_tokens": saved_tokens
    }
# --------------------

def file_line_generator(data_files):
    for file_path in data_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                yield line

def build_smart_dataset_local(
    data_files: list,
    tokenizer_path: str,
    base_tokenizer_path: str,
    output_path: str,
    k_coverage: int = 50,
    target_gb: float = 0.0,
    max_docs_to_scan: int = 5000000,
    truncate_tokens: int = 512,
    max_chars: int = 40000,
    max_tokens: int = 4096,
    num_proc: int = 16
):
    print(f"Loading new tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    first_diff_id = get_first_diff_id(base_tokenizer_path, tokenizer_path)
    special_ids = get_special_token_ids(tokenizer)
    
    total_vocab_size = len(tokenizer)
    target_tokens = get_new_token_ids(tokenizer, first_diff_id)
    
    num_target = len(target_tokens)
    total_target_hits = num_target * k_coverage
    print(f"Determined new tokens start at ID: {first_diff_id}")
    print(f"Number of target (new, non-special) tokens: {num_target}")
    print(f"Total target hits to collect (Coverage target): {total_target_hits}")
    
    target_bytes = int(target_gb * 1024 * 1024 * 1024) if target_gb > 0 else float('inf')
    
    token_counts_global = {t: 0 for t in target_tokens}
    unmet_tokens = set(target_tokens)
    
    saved_docs_count = 0
    scanned_docs = 0
    total_scanned_tokens = 0
    total_saved_tokens = 0
    current_bytes = 0
    current_hits = 0
    tokens_zero = num_target
    
    print(f"Starting multiprocessing pool with {num_proc} workers...")
    
    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write("[\n")
        
        pbar_coverage = tqdm(total=total_target_hits, desc="Token K-Coverage", unit="hits")
        
        with mp.Pool(
            processes=num_proc, 
            initializer=init_worker, 
            initargs=(tokenizer_path, first_diff_id, special_ids, max_chars, max_tokens, truncate_tokens)
        ) as pool:
            
            line_gen = file_line_generator(data_files)
            
            # imap_unordered is perfect here: it streams items in and out, keeping memory low
            for result in pool.imap_unordered(process_doc_worker, line_gen, chunksize=100):
                scanned_docs += 1
                
                if scanned_docs >= max_docs_to_scan:
                    break
                    
                if result is None:
                    continue
                    
                total_scanned_tokens += result["scanned_tokens"]
                    
                if not result["valid"]:
                    continue
                    
                counts = result["counts"]
                
                # Check if this document contains any token we STILL need
                doc_has_useful = False
                for t_id in counts.keys():
                    if t_id in unmet_tokens:
                        doc_has_useful = True
                        break
                        
                if doc_has_useful:
                    text = result["text"]
                    text_bytes = len(text.encode('utf-8'))
                    
                    if saved_docs_count > 0:
                        out_f.write(",\n")
                    out_f.write(json.dumps({"text": text}, ensure_ascii=False))
                    
                    saved_docs_count += 1
                    total_saved_tokens += result["saved_tokens"]
                    current_bytes += text_bytes
                    
                    hits_added = 0
                    for t_id, count in counts.items():
                        if t_id in unmet_tokens:
                            if token_counts_global[t_id] == 0:
                                tokens_zero -= 1
                                
                            room_left = k_coverage - token_counts_global[t_id]
                            added = min(count, room_left)
                            
                            token_counts_global[t_id] += added
                            hits_added += added
                            
                            if token_counts_global[t_id] >= k_coverage:
                                unmet_tokens.remove(t_id)
                                
                    current_hits += hits_added
                    pbar_coverage.update(hits_added)
                    
                    pbar_coverage.set_postfix({
                        "scan_d": scanned_docs,
                        "save_d": saved_docs_count,
                        "scan_t": f"{total_scanned_tokens/1e6:.2f}M",
                        "save_t": f"{total_saved_tokens/1e6:.2f}M",
                        "gb": f"{current_bytes / (1024**3):.3f}",
                        "rem": len(unmet_tokens),
                        "zero": tokens_zero
                    })
                    
                if current_bytes >= target_bytes or len(unmet_tokens) == 0:
                    break
                    
        out_f.write("\n]")
        pbar_coverage.close()
        
    print("\n--- СТАТИСТИКА СБОРА ---")
    print(f"Просканировано документов: {scanned_docs}")
    print(f"Сохранено документов: {saved_docs_count}")
    print(f"Итоговый размер: {current_bytes / (1024**3):.4f} GB")
    print(f"Всего просканировано токенов: {total_scanned_tokens:,}")
    print(f"Всего сохранено токенов: {total_saved_tokens:,}")
    
    tokens_zero = sum(1 for c in token_counts_global.values() if c == 0)
    tokens_met = sum(1 for c in token_counts_global.values() if c >= k_coverage)
    tokens_partial = len(target_tokens) - tokens_zero - tokens_met
    
    print(f"Токенов с 0 вхождений: {tokens_zero} (не встретились вообще)")
    print(f"Токенов с 1 до {k_coverage-1} вхождений: {tokens_partial}")
    print(f"Токенов, достигших квоты {k_coverage}: {tokens_met}")
    print("------------------------\n")
    print(f"Success! Saved {saved_docs_count} docs to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--base_tokenizer_path", type=str, default="/workdir/models/Qwen3.5-2B-Base")
    parser.add_argument("--k_coverage", type=int, default=50)
    parser.add_argument("--target_gb", type=float, default=0.0)
    parser.add_argument("--max_docs_to_scan", type=int, default=50000000) # Increased default
    parser.add_argument("--truncate_tokens", type=int, default=512)
    parser.add_argument("--max_chars", type=int, default=40000)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--num_proc", type=int, default=16, help="Number of CPU cores")
    
    parser.add_argument("--data_files", nargs='+', default=[
        "/shared/data/data/pre-train/darulm_25_03_25/train_part1.json",
        "/shared/data/data/pre-train/darulm_25_03_25/train_part2.json",
        "/shared/data/data/pre-train/darulm_25_03_25/train_test.json"
    ])
    
    args = parser.parse_args()
    build_smart_dataset_local(
        data_files=args.data_files,
        tokenizer_path=args.tokenizer_path,
        base_tokenizer_path=args.base_tokenizer_path,
        output_path=args.output_path,
        k_coverage=args.k_coverage,
        target_gb=args.target_gb,
        max_docs_to_scan=args.max_docs_to_scan,
        truncate_tokens=args.truncate_tokens,
        max_chars=args.max_chars,
        max_tokens=args.max_tokens,
        num_proc=args.num_proc
    )
