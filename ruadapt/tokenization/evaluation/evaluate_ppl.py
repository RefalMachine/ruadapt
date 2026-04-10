import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def evaluate_ppl_core(model_path, data_path, num_docs=1000, max_tokens=512):
    print(f"Loading model and tokenizer from: {model_path}")
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path)
    
    architectures = getattr(config, "architectures", [])
    if architectures and "Qwen3_5ForConditionalGeneration" in architectures:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
        ModelClass = Qwen3_5ForConditionalGeneration
    else:
        ModelClass = AutoModelForCausalLM

    model = ModelClass.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )
    model.eval()

    print(f"Loading data from: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Ожидаем, что data это список словарей с ключом 'text' (или просто строки)
    docs = []
    for item in data:
        if isinstance(item, dict) and 'text' in item:
            docs.append(item['text'])
        elif isinstance(item, str):
            docs.append(item)
    
    docs = docs[:num_docs]
    print(f"Evaluating on {len(docs)} documents (max_tokens={max_tokens})...")

    nlls = []
    
    with torch.no_grad():
        for text in tqdm(docs, desc="Calculating PPL"):
            # Токенизируем с ограничением по длине
            encodings = tokenizer(
                text,
                max_length=max_tokens,
                truncation=True,
                return_tensors="pt"
            )
            assert len(encodings.input_ids) <= max_tokens
            input_ids = encodings.input_ids.to(model.device)
            target_ids = input_ids.clone()

            # Если документ оказался слишком коротким (меньше 2 токенов), пропускаем
            if input_ids.shape[1] < 2:
                continue

            outputs = model(input_ids, labels=target_ids)
            
            # Извлекаем скаляр и удаляем граф, чтобы избежать утечек памяти (особенно в кастомных VLM)
            loss_val = outputs.loss.item()
            neg_log_likelihood = loss_val * input_ids.shape[1]
            nlls.append(neg_log_likelihood)
            
            del outputs, encodings, input_ids, target_ids
            torch.cuda.empty_cache()

    if not nlls:
        print("Error: No valid documents to evaluate.")
        return None, 0

    # Считаем точную перплексию по корпусу
    total_tokens = sum(min(max_tokens, len(tokenizer(doc)['input_ids'])) for doc in docs)
    total_nll = sum(nlls)
    corpus_ppl = torch.exp(torch.tensor(total_nll) / total_tokens).item()

    return corpus_ppl, total_tokens

def evaluate_ppl(model_path, data_path, num_docs=1000, max_tokens=512):
    corpus_ppl, total_tokens = evaluate_ppl_core(model_path, data_path, num_docs, max_tokens)
    if corpus_ppl is None:
        return

    print("\n--- Results ---")
    print(f"Model: {model_path}")
    print(f"Total tokens: {total_tokens}")
    print(f"Corpus Perplexity: {corpus_ppl:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to initialized model")
    parser.add_argument('--data_path', type=str, default="data/rus.json", help="Path to json file with texts")
    parser.add_argument('--num_docs', type=int, default=1000, help="Number of documents to process")
    parser.add_argument('--max_tokens', type=int, default=512, help="Max tokens per document")
    args = parser.parse_args()

    evaluate_ppl(args.model_path, args.data_path, args.num_docs, args.max_tokens)
