import json
import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from ruadapt.tokenization.utils import get_first_diff_id, get_special_token_ids
from ruadapt.initialization.data.streamer import FilteredDatasetStreamer

def build_k_coverage_dataset(
    tokenizer_path: str,
    base_tokenizer_path: str,
    output_path: str,
    k_coverage: int = 50,
    max_docs_to_save: int = 0,
    target_gb: float = 0.0,
    max_docs_to_scan: int = 5000000,
    dataset_name: str = "HuggingFaceFW/fineweb-2",
    subset: str = "rus_Cyrl",
    batch_size: int = 1000,
    truncate_tokens: int = 512,
    max_chars: int = 40000,
    max_tokens: int = 4096
):
    print(f"Loading new tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    first_diff_id = get_first_diff_id(base_tokenizer_path, tokenizer_path)
    print(f"Determined new tokens start at ID: {first_diff_id}")
    
    total_vocab_size = len(tokenizer)
    print(f"Total vocab size: {total_vocab_size}")
    
    target_tokens = set(range(first_diff_id, total_vocab_size))
    
    # Исключаем всевозможные специальные токены
    special_ids = get_special_token_ids(tokenizer)
                
    if special_ids:
        print(f"Found {len(special_ids)} special tokens across all vocabularies. Removing them from targets...")
        target_tokens = target_tokens - special_ids
        
    num_target = len(target_tokens)
    print(f"Number of target (new, non-special) tokens: {num_target}")
    
    # Трекинг
    token_counts = {t: 0 for t in target_tokens}
    unmet_tokens = set(target_tokens) # Токены, у которых счетчик < k_coverage
    
    total_target_hits = num_target * k_coverage
    print(f"Total target hits to collect (Coverage target): {total_target_hits}")
    
    import time
    
    time_coverage = 0.0
    time_io = 0.0
    
    print(f"Streaming dataset {dataset_name} ({subset}) with length filtering (max {max_docs_to_scan} scan)...")
    # Tokenization is NOT omitted here because we need the input_ids to check coverage targets anyway
    # We apply truncation=512 for fast evaluation pipelines
    streamer = FilteredDatasetStreamer(
        tokenizer=tokenizer, 
        dataset_name=dataset_name, 
        subset=subset, 
        batch_size=batch_size,
        truncate_tokens=truncate_tokens,
        max_chars=max_chars,
        max_tokens=max_tokens
    )
    
    target_bytes = int(target_gb * 1024 * 1024 * 1024) if target_gb > 0 else float('inf')
    max_docs = max_docs_to_save if max_docs_to_save > 0 else float('inf')
    
    if target_bytes == float('inf') and max_docs == float('inf'):
        print("WARNING: Neither target_gb nor max_docs_to_save specified. Will run until k_coverage is met or max_docs_to_scan reached.")
        
    saved_docs_count = 0
    scanned_docs = 0
    current_bytes = 0
    current_hits = 0
    
    print(f"Streaming smart dataset directly to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write("[\n")
        
        # Главный прогресс-бар по покрытию квоты
        pbar_coverage = tqdm(total=total_target_hits, desc="Token K-Coverage", unit="hits")
        
        for valid_batch in streamer.stream_batches():
            if scanned_docs >= max_docs_to_scan or saved_docs_count >= max_docs or current_bytes >= target_bytes or len(unmet_tokens) == 0:
                break
                
            scanned_docs += streamer.raw_scanned - scanned_docs # Rough approximation since streamer reads ahead
            
            for doc in valid_batch:
                text = doc["text"]
                input_ids = doc["input_ids"]
                
                t0 = time.time()
                doc_tokens_set = set(input_ids)
                
                # Проверяем, есть ли в документе нужные нам токены, квота по которым еще не выполнена
                useful_tokens = doc_tokens_set & unmet_tokens
                time_coverage += (time.time() - t0)
                
                if useful_tokens:
                    t0 = time.time()
                    # Если мы обрезали токены, текст должен соответствовать обрезанным токенам
                    # Для экономии места и точности мы декодируем обрезанные токены обратно в текст
                    if streamer.truncate_tokens > 0 and len(input_ids) >= streamer.truncate_tokens:
                        text = tokenizer.decode(input_ids)
                        
                    text_bytes = len(text.encode('utf-8'))
                    
                    # Документ полезен! Пишем на диск.
                    if saved_docs_count > 0:
                        out_f.write(",\n")
                    out_f.write(json.dumps({"text": text}, ensure_ascii=False))
                    
                    saved_docs_count += 1
                    current_bytes += text_bytes
                    time_io += (time.time() - t0)
                    
                    # Обновляем счетчики
                    t0 = time.time()
                    hits_added = 0
                    for t in useful_tokens:
                        token_counts[t] += 1
                        hits_added += 1
                        if token_counts[t] >= k_coverage:
                            unmet_tokens.remove(t)
                            
                    current_hits += hits_added
                    pbar_coverage.update(hits_added)
                    time_coverage += (time.time() - t0)
                    
            # Обновляем инфу сбоку ПОСЛЕ ОБРАБОТКИ БАТЧА
            pbar_coverage.set_postfix({
                "raw_scan": streamer.raw_scanned,
                "saved": saved_docs_count,
                "gb": f"{current_bytes / (1024**3):.4f}",
                "rem": len(unmet_tokens)
            })
                
        out_f.write("\n]")
                
    pbar_coverage.close()
    streamer.print_stats()
    
    print(f"\n--- ТАЙМИНГИ SMART ЛОГИКИ ---")
    print(f"Подсчет K-Coverage (Sets & Counters): {time_coverage:.2f} сек")
    print(f"Запись на диск (JSON Dumps): {time_io:.2f} сек")
    print("--------------------------------------\n")
    
    print("\n--- СТАТИСТИКА СБОРА ---")
    print(f"Просканировано документов: {scanned_docs}")
    print(f"Сохранено документов: {saved_docs_count}")
    
    tokens_zero = sum(1 for c in token_counts.values() if c == 0)
    tokens_met = sum(1 for c in token_counts.values() if c >= k_coverage)
    tokens_partial = len(target_tokens) - tokens_zero - tokens_met
    
    print(f"Токенов с 0 вхождений: {tokens_zero} (не встретились вообще)")
    print(f"Токенов с 1 до {k_coverage-1} вхождений: {tokens_partial}")
    print(f"Токенов, достигших квоты {k_coverage}: {tokens_met}")
    print("------------------------\n")
    print(f"Success! Saved {saved_docs_count} docs to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the extended tokenizer")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save the JSON dataset")
    parser.add_argument("--base_tokenizer_path", type=str, default="/workdir/models/Qwen3.5-2B-Base", help="Path to the original tokenizer to find diff")
    parser.add_argument("--k_coverage", type=int, default=50, help="Target frequency for each new token")
    parser.add_argument("--max_docs_to_save", type=int, default=0, help="Max documents to include (0 to ignore)")
    parser.add_argument("--target_gb", type=float, default=0.0, help="Target dataset size in GB (0 to ignore)")
    parser.add_argument("--max_docs_to_scan", type=int, default=5000000, help="Max documents to read from corpus")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--truncate_tokens", type=int, default=512, help="Truncate documents to N tokens (0 or -1 to disable)")
    parser.add_argument("--max_chars", type=int, default=40000, help="Max characters per document before tokenization")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens per document after tokenization")
    
    args = parser.parse_args()
    build_k_coverage_dataset(
        tokenizer_path=args.tokenizer_path,
        base_tokenizer_path=args.base_tokenizer_path,
        output_path=args.output_path,
        k_coverage=args.k_coverage,
        max_docs_to_save=args.max_docs_to_save,
        target_gb=args.target_gb,
        max_docs_to_scan=args.max_docs_to_scan,
        batch_size=args.batch_size,
        truncate_tokens=args.truncate_tokens,
        max_chars=args.max_chars,
        max_tokens=args.max_tokens
    )
    
    # Workaround for HuggingFace datasets PyGILState_Release bug on exit
    import os
    os._exit(0)
