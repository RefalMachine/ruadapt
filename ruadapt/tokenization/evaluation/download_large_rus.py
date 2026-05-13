import os
import json
from datasets import load_dataset

def download_fineweb_large(output_path, max_docs=20000):
    print(f"Downloading {max_docs} docs for Russian from HuggingFaceFW/fineweb-2...")
    dataset = load_dataset("HuggingFaceFW/fineweb-2", name="rus_Cyrl", split="train", streaming=True)
    
    docs = []
    for i, doc in enumerate(dataset):
        if i >= max_docs:
            break
        docs.append({"id": doc.get("id", str(i)), "text": doc["text"]})
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(docs)} documents to {output_path}")

if __name__ == "__main__":
    download_fineweb_large("/workdir/projects/tokenizer_init_research/ruadapt/ruadapt/tokenization/evaluation/data/rus_large.json", max_docs=20000)
