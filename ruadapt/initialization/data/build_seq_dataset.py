import argparse
import os
import json
import time
import multiprocessing as mp
from tqdm import tqdm
from transformers import AutoTokenizer

from ruadapt.tokenization.utils import get_first_diff_id, get_special_token_ids

# --- WORKER LOGIC ---
WORKER_TOKENIZER = None
WORKER_MAX_CHARS = None
WORKER_MAX_TOKENS = None
WORKER_TRUNCATE = None

def init_worker(tokenizer_path, max_chars, max_tokens, truncate_tokens):
    global WORKER_TOKENIZER, WORKER_MAX_CHARS, WORKER_MAX_TOKENS, WORKER_TRUNCATE
    WORKER_TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path)
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
    
    if len(input_ids) > WORKER_MAX_TOKENS:
        return None
        
    if WORKER_TRUNCATE > 0:
        input_ids = input_ids[:WORKER_TRUNCATE]
        
    if WORKER_TRUNCATE > 0 and len(input_ids) >= WORKER_TRUNCATE:
        text = WORKER_TOKENIZER.decode(input_ids)
        
    return {"text": text}
# --------------------

def file_line_generator(data_files):
    for file_path in data_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                yield line

def build_seq_dataset_local(
    data_files: list,
    tokenizer_path: str,
    output_path: str,
    target_gb: float = 1.0,
    truncate_tokens: int = 512,
    max_chars: int = 40000,
    max_tokens: int = 4096,
    num_proc: int = 16
):
    print(f"Loading tokenizer from {tokenizer_path}...")
    
    target_bytes = target_gb * 1024 * 1024 * 1024
    print(f"Target size: {target_gb} GB ({target_bytes} bytes)")
    
    saved_docs_count = 0
    scanned_docs = 0
    current_bytes = 0
    
    print(f"Starting multiprocessing pool with {num_proc} workers...")
    
    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write("[\n")
        
        pbar_bytes = tqdm(total=target_bytes, desc="Saved (Bytes)", unit="B", unit_scale=True)
        
        with mp.Pool(
            processes=num_proc, 
            initializer=init_worker, 
            initargs=(tokenizer_path, max_chars, max_tokens, truncate_tokens)
        ) as pool:
            
            line_gen = file_line_generator(data_files)
            
            for result in pool.imap_unordered(process_doc_worker, line_gen, chunksize=100):
                scanned_docs += 1
                
                if result is None:
                    continue
                    
                text = result["text"]
                text_bytes = len(text.encode('utf-8'))
                
                if saved_docs_count > 0:
                    out_f.write(",\n")
                out_f.write(json.dumps({"text": text}, ensure_ascii=False))
                
                saved_docs_count += 1
                current_bytes += text_bytes
                
                pbar_bytes.update(text_bytes)
                pbar_bytes.set_postfix({"scan": scanned_docs, "saved": saved_docs_count})
                
                if current_bytes >= target_bytes:
                    break
                    
        out_f.write("\n]")
        pbar_bytes.close()
        
    print("\n--- СТАТИСТИКА СБОРА ---")
    print(f"Просканировано документов: {scanned_docs}")
    print(f"Сохранено документов: {saved_docs_count}")
    print(f"Итоговый размер: {current_bytes / (1024**3):.4f} GB")
    print("------------------------\n")
    print(f"Success! Saved {saved_docs_count} docs to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--target_gb", type=float, required=True)
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
    build_seq_dataset_local(
        data_files=args.data_files,
        tokenizer_path=args.tokenizer_path,
        output_path=args.output_path,
        target_gb=args.target_gb,
        truncate_tokens=args.truncate_tokens,
        max_chars=args.max_chars,
        max_tokens=args.max_tokens,
        num_proc=args.num_proc
    )
