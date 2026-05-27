import time
from datasets import load_dataset
from typing import Iterator, Dict, Any, List

class FilteredDatasetStreamer:
    """
    Универсальный стример датасета.
    Фильтрует документы по количеству символов (до токенизации) 
    и по количеству токенов (после токенизации).
    Выдает батчи проверенных документов для оптимальной работы.
    """
    def __init__(
        self, 
        tokenizer=None,
        dataset_name: str = "HuggingFaceFW/fineweb-2", 
        subset: str = "rus_Cyrl",
        max_chars: int = 40000,
        max_tokens: int = 4096,
        batch_size: int = 1000,
        omit_tokenization: bool = False,
        truncate_tokens: int = -1
    ):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.subset = subset
        self.max_chars = max_chars
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.omit_tokenization = omit_tokenization
        self.truncate_tokens = truncate_tokens
        
        # Статистика отбраковки
        self.raw_scanned = 0
        self.rejected_by_chars = 0
        self.rejected_by_tokens = 0
        
        # Тайминги профилирования
        self.time_download = 0.0
        self.time_tokenize = 0.0
        self.time_filter = 0.0
        
    def stream_batches(self) -> Iterator[List[Dict[str, Any]]]:
        dataset = load_dataset(self.dataset_name, name=self.subset, split="train", streaming=True)
        doc_iterator = iter(dataset)
        
        while True:
            # 1. Собираем батч, проходящий фильтр по символам
            t0 = time.time()
            batch_texts = []
            while len(batch_texts) < self.batch_size:
                try:
                    doc = next(doc_iterator)
                    self.raw_scanned += 1
                except StopIteration:
                    break
                
                text = doc["text"]
                if len(text) > self.max_chars:
                    self.rejected_by_chars += 1
                    continue
                    
                batch_texts.append(text)
            self.time_download += (time.time() - t0)
                
            if not batch_texts:
                break
                
            valid_batch = []
            
            # Если токенизация не нужна (или токенизатора нет), просто возвращаем тексты
            if self.omit_tokenization or self.tokenizer is None:
                t0 = time.time()
                for text in batch_texts:
                    valid_batch.append({
                        "text": text,
                        "input_ids": None
                    })
                self.time_filter += (time.time() - t0)
            else:
                # 2. Токенизируем весь батч разом (это быстро)
                t0 = time.time()
                encodings = self.tokenizer(batch_texts, add_special_tokens=False)
                self.time_tokenize += (time.time() - t0)
                
                # 3. Фильтруем и обрезаем (truncate) по количеству токенов
                t0 = time.time()
                for text, input_ids in zip(batch_texts, encodings["input_ids"]):
                    if len(input_ids) > self.max_tokens:
                        self.rejected_by_tokens += 1
                        continue
                        
                    if self.truncate_tokens > 0:
                        input_ids = input_ids[:self.truncate_tokens]
                        # Текст мы НЕ обрезаем на лету (т.к. detokenize дорогой), 
                        # мы будем использовать обрезанные input_ids для покрытия
                        
                    valid_batch.append({
                        "text": text,
                        "input_ids": input_ids
                    })
                self.time_filter += (time.time() - t0)
                    
            if valid_batch:
                yield valid_batch
                
    def print_stats(self):
        print("\n--- СТАТИСТИКА ФИЛЬТРАЦИИ СТРИМЕРА ---")
        print(f"Всего просмотрено (raw): {self.raw_scanned}")
        print(f"Отброшено по длине символов (>{self.max_chars}): {self.rejected_by_chars}")
        print(f"Отброшено по длине токенов (>{self.max_tokens}): {self.rejected_by_tokens}")
        valid = self.raw_scanned - self.rejected_by_chars - self.rejected_by_tokens
        pct = (valid / self.raw_scanned * 100) if self.raw_scanned else 0
        print(f"Пропущено дальше: {valid} ({pct:.1f}%)")
        print("\n--- ТАЙМИНГИ СТРИМЕРА ---")
        print(f"Скачивание (HF Download): {self.time_download:.2f} сек")
        print(f"Токенизация (HF Tokenizer): {self.time_tokenize:.2f} сек")
        print(f"Фильтрация (Python): {self.time_filter:.2f} сек")
        print("--------------------------------------\n")
