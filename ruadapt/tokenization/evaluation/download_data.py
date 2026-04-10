import argparse
import os
import json
from datasets import load_dataset
from tqdm import tqdm

def download_fineweb(name, subset, output_path, max_docs=1000, dataset_name="HuggingFaceFW/fineweb-2"):
    print(f"Downloading {max_docs} docs for {name} ({subset}) from {dataset_name}...")
    dataset = load_dataset(dataset_name, name=subset, split="train", streaming=True)
    
    docs = []
    for i, doc in enumerate(dataset):
        if i >= max_docs:
            break
        docs.append({"id": doc.get("id", str(i)), "text": doc["text"]})
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print(f"Saved to {output_path}")

def main():
    languages = [
        ("eng", "CC-MAIN-2024-10", "HuggingFaceFW/fineweb"),
        ("rus", "rus_Cyrl", "HuggingFaceFW/fineweb-2"),
        ("cmn", "cmn_Hani", "HuggingFaceFW/fineweb-2"),
        ("deu", "deu_Latn", "HuggingFaceFW/fineweb-2"),
        ("spa", "spa_Latn", "HuggingFaceFW/fineweb-2"),
        ("fra", "fra_Latn", "HuggingFaceFW/fineweb-2"),
        ("ita", "ita_Latn", "HuggingFaceFW/fineweb-2"),
        ("arb", "arb_Arab", "HuggingFaceFW/fineweb-2"),
        ("jpn", "jpn_Jpan", "HuggingFaceFW/fineweb-2"),
        ("hin", "hin_Deva", "HuggingFaceFW/fineweb-2"),
        ("ukr", "ukr_Cyrl", "HuggingFaceFW/fineweb-2"),
        ("kaz", "kaz_Cyrl", "HuggingFaceFW/fineweb-2")
    ]
    
    os.makedirs("/workdir/projects/tokenizer_extension/evaluation/data", exist_ok=True)
    
    for lang, subset, dataset_name in languages:
        output_path = f"/workdir/projects/tokenizer_extension/evaluation/data/{lang}.json"
        if not os.path.exists(output_path):
            download_fineweb(lang, subset, output_path, max_docs=1000, dataset_name=dataset_name)
        else:
            print(f"File {output_path} already exists, skipping.")

if __name__ == "__main__":
    main()
