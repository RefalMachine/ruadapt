import argparse
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def evaluate_ppl_core(model_path, data_path, num_docs=1000, max_tokens=2048, batch_size=2):
    print(f"Loading model and tokenizer from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    architectures = getattr(config, "architectures", [])
    if architectures and "Qwen3_5ForConditionalGeneration" in architectures:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
        ModelClass = Qwen3_5ForConditionalGeneration
    else:
        ModelClass = AutoModelForCausalLM
    ModelClass = AutoModelForCausalLM
    model = ModelClass.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    model.eval()

    print(f"Loading data from: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    docs = []
    for item in data:
        if isinstance(item, dict) and 'text' in item:
            docs.append(item['text'])
        elif isinstance(item, str):
            docs.append(item)
    
    docs = docs[:num_docs]
    print(f"Evaluating on {len(docs)} documents (max_tokens={max_tokens}, batch_size={batch_size})...")

    # 1. Токенизация всего батча разом (Fast Rust Backend)
    print(f"Batch tokenizing documents (truncating to {max_tokens} tokens)...")
    encodings = tokenizer(
        docs,
        max_length=max_tokens,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )

    dataset = torch.utils.data.TensorDataset(encodings.input_ids, encodings.attention_mask)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_nll = 0.0
    total_tokens_count = 0
    
    # 2. Быстрый проход через GPU (батчами, без чанков)
    with torch.no_grad():
        for batch_input_ids, batch_attention_mask in tqdm(dataloader, desc="Calculating PPL"):
            batch_input_ids = batch_input_ids.to(model.device)
            batch_attention_mask = batch_attention_mask.to(model.device)

            # CausalLM игнорирует индексы -100 при расчете Loss
            labels = batch_input_ids.clone()
            labels[batch_attention_mask == 0] = -100

            # HuggingFace автоматически делает shift (labels[..., 1:]) и считает loss
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                labels=labels
            )
            
            # Loss возвращается уже усредненным по количеству валидных (не -100) токенов
            loss = outputs.loss
            
            # Считаем количество валидных токенов в этом батче (учитывая сдвиг)
            shifted_labels = labels[..., 1:]
            valid_tokens = (shifted_labels != -100).sum().item()

            if valid_tokens > 0:
                # Накапливаем сумму логарифмов (NLL) для точного расчета PPL по всему корпусу
                total_nll += loss.item() * valid_tokens
                total_tokens_count += valid_tokens
                
    if total_tokens_count == 0:
        print("Error: No valid documents to evaluate.")
        return None, 0

    corpus_ppl = torch.exp(torch.tensor(total_nll) / total_tokens_count).item()
    return corpus_ppl, total_tokens_count


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
