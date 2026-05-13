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
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    model.eval()
    print(model.get_input_embeddings().weight[260577])
    print(model.get_output_embeddings().weight[260577])
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
    total_tokens_count = 0
    
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


            
            seq_len = input_ids.shape[1]
            
            # -------------------------------------------------------------
            # MEMORY OPTIMIZATION: Chunked LM Head computation
            # To avoid [SeqLen, 248320] VRAM spike, we run base_model
            # to get hidden states, then chunk the lm_head and cross-entropy
            # -------------------------------------------------------------
            base_model = getattr(model, 'model', getattr(model, 'transformer', getattr(model, 'base_model', None)))
            lm_head = model.get_output_embeddings()
            if lm_head is None:
                lm_head = getattr(model, 'lm_head', None)
                
            base_outputs = base_model(input_ids)
            hidden_states = base_outputs[0] # [1, SeqLen, H]
            
            # Shift for causal LM (predict next token)
            shift_hidden_states = hidden_states[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            total_chunk_loss = 0.0
            chunk_size = 512 # small enough to fit in any VRAM
            debug_prints = getattr(evaluate_ppl_core, "debug_prints", 0) # Store across docs
            
            for i in range(0, shift_hidden_states.size(1), chunk_size):
                h_chunk = shift_hidden_states[:, i:i+chunk_size, :]
                labels_chunk = shift_labels[:, i:i+chunk_size].view(-1)
                
                # Compute logits just for this chunk
                logits_chunk = lm_head(h_chunk).float()
                flat_logits = logits_chunk.view(-1, logits_chunk.size(-1))
                
                # Compute individual losses
                loss_chunk = torch.nn.functional.cross_entropy(
                    flat_logits, 
                    labels_chunk, 
                    reduction='none'
                )
                total_chunk_loss += loss_chunk.sum().item()
                
                if debug_prints < 10:
                    high_loss_indices = (loss_chunk > 10.0).nonzero(as_tuple=True)[0]
                    for idx in high_loss_indices:
                        if debug_prints >= 10:
                            break
                        
                        actual_id = labels_chunk[idx].item()
                        # Avoid decoding special padding or ignore indices if any (-100)
                        if actual_id != -100:
                            actual_token = tokenizer.decode([actual_id])
                            actual_loss = loss_chunk[idx].item()
                            
                            topk_logits, topk_ids = torch.topk(flat_logits[idx], k=5)
                            topk_tokens = [tokenizer.decode([tid.item()]) for tid in topk_ids]
                            
                            print(f"\n[DEBUG] High Loss: {actual_loss:.2f} on token '{actual_token}' (ID: {actual_id})")
                            print(f"Top 5 predictions:")
                            for rank, (tid, ttok, tlog) in enumerate(zip(topk_ids, topk_tokens, topk_logits)):
                                print(f"  {rank+1}. '{ttok}' (ID: {tid.item()}) - logit: {tlog.item():.2f}")
                            debug_prints += 1
                evaluate_ppl_core.debug_prints = debug_prints
                
                del logits_chunk, flat_logits, h_chunk, labels_chunk, loss_chunk
                
            # Average over shifted seq len (N-1)
            loss_val = total_chunk_loss / shift_labels.size(1)
            
            # Multiply by original seq_len to match standard HF perplexity log formulation
            neg_log_likelihood = loss_val * seq_len
            nlls.append(neg_log_likelihood)
            total_tokens_count += seq_len
            
            # Clean up properly
            del base_outputs
            del hidden_states
            del shift_hidden_states
            del shift_labels
            del encodings
            del input_ids
            del target_ids
            
            torch.cuda.empty_cache()

    if not nlls:
        print("Error: No valid documents to evaluate.")
        return None, 0

    # Считаем точную перплексию по корпусу
    total_nll = sum(nlls)
    corpus_ppl = torch.exp(torch.tensor(total_nll) / total_tokens_count).item()
    total_tokens = total_tokens_count

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
