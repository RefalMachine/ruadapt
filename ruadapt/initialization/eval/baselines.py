import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from ruadapt.initialization.head.metrics import FastMetrics

def compute_metrics(pred, target_ids, fast_metrics):
    # Using the centralized FastMetrics logic
    m_dict = fast_metrics.compute_batch_metrics(pred, target_ids)
    
    # To maintain bucket granularity in the script, we need individual values.
    # We will reconstruct just what we need per batch item since compute_batch_metrics returns means.
    
    targets = fast_metrics.embeddings[target_ids]
    mse_dist = torch.nn.functional.mse_loss(pred, targets, reduction='none').mean(dim=1)
    cos_sim = torch.nn.functional.cosine_similarity(pred, targets, dim=-1)
    cos_dist = 1.0 - cos_sim
    
    dist_matrix = fast_metrics.compute_distance_matrix(pred)
    B = pred.size(0)
    batch_idx = torch.arange(B, device=pred.device)
    d_target = dist_matrix[batch_idx, target_ids].clone()
    
    # MRR
    ranks = (dist_matrix < d_target.unsqueeze(1)).sum(dim=1) + 1
    mrr = 1.0 / ranks.float()
    
    # RelDist
    dist_matrix[batch_idx, target_ids] = float('inf')
    d_nearest, _ = dist_matrix.min(dim=1)
    rel_dist = d_target / (d_target + d_nearest + 1e-9)
    
    return mse_dist, cos_dist, rel_dist, mrr

def evaluate(dataset, embeddings, fast_metrics, method="mean", best_only=False):
    
    # Group by target_id
    grouped = {}
    for item in dataset:
        tid = item["target_id"]
        if tid not in grouped:
            grouped[tid] = []
        grouped[tid].append(item)
        
    all_preds, all_tids, all_lengths, all_masks = [], [], [], []
    
    for tid, items in grouped.items():
        if best_only:
            len2_items = [x for x in items if len(x["fragmented_ids"]) == 2]
            if not len2_items: continue
            best_item = max(len2_items, key=lambda x: x.get("count", 0))
            splits_to_eval = [best_item["fragmented_ids"]]
        else:
            splits_to_eval = [x["fragmented_ids"] for x in items]
            
        for frag_ids in splits_to_eval:
            frag_vecs = embeddings[frag_ids].float()
            if method == "mean":
                pred_vec = frag_vecs.mean(dim=0)
            elif method == "weighted_mean":
                decay = 0.8
                weights = torch.tensor([decay ** i for i in range(len(frag_ids))], device=frag_vecs.device).float()
                weights = weights / weights.sum()
                pred_vec = (frag_vecs * weights.unsqueeze(1)).sum(dim=0)
                
            all_preds.append(pred_vec)
            all_tids.append(tid)
            all_lengths.append(len(frag_ids))
            
            # Rule: mask sub-tokens ONLY if we average identical tokens
            mask_idx = [tid]
            if len(set(frag_ids)) == 1:
                mask_idx.append(frag_ids[0])
            all_masks.append(mask_idx)
            
    if not all_preds:
        return {}

    all_preds = torch.stack(all_preds)
    all_tids = torch.tensor(all_tids, device=embeddings.device)
    
    # Batch process
    batch_size = 512
    bucket_metrics = {2: {'mse':[], 'cos':[], 'rel_dist':[], 'mrr':[]}, 
                      3: {'mse':[], 'cos':[], 'rel_dist':[], 'mrr':[]}, 
                      '4+': {'mse':[], 'cos':[], 'rel_dist':[], 'mrr':[]}}
                      
    for i in tqdm(range(0, len(all_preds), batch_size), desc=f"Eval {method}"):
        b_preds = all_preds[i:i+batch_size]
        b_tids = all_tids[i:i+batch_size]
        b_len = all_lengths[i:i+batch_size]
        b_masks = all_masks[i:i+batch_size]
        
        mse, cos, rel_dist, mrr = compute_metrics(b_preds, b_tids, fast_metrics)
        
        for j, l in enumerate(b_len):
            bucket = 2 if l == 2 else (3 if l == 3 else '4+')
            bucket_metrics[bucket]['mse'].append(mse[j].item())
            bucket_metrics[bucket]['cos'].append(cos[j].item())
            bucket_metrics[bucket]['rel_dist'].append(rel_dist[j].item())
            bucket_metrics[bucket]['mrr'].append(mrr[j].item())
            
    return bucket_metrics

def print_metrics(bucket_metrics, method):
    print(f"\n{method} Metrics Breakdown:")
    total_mse, total_cos, total_rel_dist, total_mrr, total_count = 0, 0, 0, 0, 0
    
    for bucket in [2, 3, '4+']:
        metrics = bucket_metrics[bucket]
        count = len(metrics['mse'])
        if count > 0:
            avg_mse = sum(metrics['mse']) / count
            avg_cos = sum(metrics['cos']) / count
            avg_rel = sum(metrics['rel_dist']) / count
            avg_mrr = sum(metrics['mrr']) / count
            
            total_mse += sum(metrics['mse'])
            total_cos += sum(metrics['cos'])
            total_rel_dist += sum(metrics['rel_dist'])
            total_mrr += sum(metrics['mrr'])
            total_count += count
            
            print(f" - Length {bucket:>2}: MSE {avg_mse:.6f} | CosDist {avg_cos:.6f} | RelDist {avg_rel:.4f} | MRR {avg_mrr:.4f} (n={count})")
            
    if total_count > 0:
        print(f" - OVERALL  : MSE {total_mse/total_count:.6f} | CosDist {total_cos/total_count:.6f} | RelDist {total_rel_dist/total_count:.4f} | MRR {total_mrr/total_count:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-only", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/workdir/models/Qwen3.5-2B-Base" # External volume
    
    print("Loading embeddings...")
    llm = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cpu")
    embeddings = llm.get_input_embeddings().weight.detach().to(device).float()
    
    with open('data/val_bpe.json', 'r') as f:
        val_data = json.load(f)
        
    print(f"\nVal size: {len(val_data)} split pairs")
    
    for method in ["mean", "weighted_mean"]:
        fast_metrics = FastMetrics(embeddings)
        metrics = evaluate(val_data, embeddings, fast_metrics, method=method, best_only=args.best_only)
        print_metrics(metrics, method)

if __name__ == "__main__":
    main()
