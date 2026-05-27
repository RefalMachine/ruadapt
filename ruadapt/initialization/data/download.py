import json
from datasets import load_dataset

print("Downloading larger dataset for Mini-CPT...")
dataset = load_dataset("HuggingFaceFW/fineweb-2", name="rus_Cyrl", split="train", streaming=True)

docs = []
for i, doc in enumerate(dataset):
    if i >= 20000:
        break
    docs.append({"text": doc["text"]})

with open("data/rus_large.json", "w", encoding="utf-8") as f:
    json.dump(docs, f, ensure_ascii=False)

print(f"Saved 20,000 docs to data/rus_large.json")
