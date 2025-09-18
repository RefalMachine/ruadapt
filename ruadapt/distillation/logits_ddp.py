import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import time
import multiprocessing as mp


def setup_distributed():
    """Initialization of a distributed group of processes."""
    if not dist.is_initialized():
        # torchrun sets MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE
        dist.init_process_group(backend='nccl')
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f"cuda:{local_rank}")
    
    torch.cuda.set_device(device)
    print(f"Rank {rank}/{world_size} started on device: {device}")
    
    return rank, world_size, device

def cleanup_distributed():
    """Cleaning up distributed group resources."""
    dist.destroy_process_group()

def read_jsonl(file_name):
    with open(file_name, encoding="utf-8") as r:
        return [json.loads(line) for line in r]

def read_json(file_name):
    with open(file_name, encoding="utf-8") as r:
        return json.load(r)
    
def write_jsonl(records, path):
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)


    with open(path, "w", encoding="utf-8") as w:
        for r in records:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_texts(input_path, tokenizer, max_len, sort_by_len):
    # Loading dataset with datasets.load_dataset
    dataset = load_dataset('json', data_files={'data': input_path})['data']
    
    # Determine the number of processes for map
    num_proc = min(mp.cpu_count(), 64)

    # Function for processing one record
    def process_record(example, idx):
        role_mapping = {
            "bot": "assistant",
            "gpt": "assistant",
            "human": "user",
        }

        if "instruction" in example:
            messages = [{"role": "user", "content": example["instruction"]}]
        elif "messages" in example or "prompt" in example or 'turns' in example:
            messages = example.get("messages", example.get("prompt", example.get('turns')))
        else:
            return None  # Skipping records without required fields

        if not messages:
            return None

        for m in messages:
            if 'role' not in m:
                if len(messages) == 1:
                    m['role'] = 'user'
                else:
                    return None  # Skip if there is no role and more than one message

        # Check that there is only one assistant
        if len([m for m in messages if m['role'] in ['assistant', 'bot']]) > 1:
            return None

        for m in messages:
            m["role"] = role_mapping.get(m["role"], m["role"])

        # Trying to tokenize
        try:
            prompt_tokens = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False
            )
            if len(prompt_tokens) >= max_len:
                return None  # Skipping overly long entries
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            # Return a dictionary as required by datasets.map
            return {"idx": idx, "text": prompt_text, "input_len": len(prompt_tokens)}
        except Exception:
            return None  # Skipping records that cause errors

    # Filter None values
    processed_dataset = dataset.filter(lambda example, idx: process_record(example, idx) is not None, with_indices=True, num_proc=num_proc, load_from_cache_file=False)
    
    # Apply processing to the entire dataset
    processed_dataset = processed_dataset.map(
        lambda example, idx: process_record(example, idx),
        with_indices=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        desc="Processing dataset"
    )

    # Convert to a list of tuples (idx, text, input_len)
    data = [(item["idx"], item["text"], item["input_len"]) for item in processed_dataset if item["input_len"] is not None]
    print(data[0])

    if sort_by_len:
        data = sorted(data, key=lambda x: -x[-1])
        
    return data
    
class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"text": self.data[idx][1], "original_index": self.data[idx][0]}
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--input_file', type=str, default='input.json')
    parser.add_argument('--output_file', type=str, default='output.json')
    parser.add_argument('--filtered_data_file', type=str, default='filtered_data.json')
    parser.add_argument('--max_length', type=int, default=8192)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--sort_by_len', action="store_true")
    parser.add_argument('--per_token', action='store_true')
    parser.add_argument('--topk', type=int, default=5)
    
    args = parser.parse_args()
    
    start = time.time()
    
    rank, world_size, device = setup_distributed()

    print(f"Rank {rank}: Loading model and tokenizer...")

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2', device_map=device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = 'left'
    model.eval()
    print(f"Rank {rank}: model loaded...")


    dist.barrier()
    print(f"Rank {rank}: barrier 1 passed...")
    
    # --- 3. Data preprocessing ---
    texts = []
    if rank == 0:
        print("Rank 0: Loading and broadcasting data...")
        texts = load_texts(args.input_file, tokenizer, args.max_length, args.sort_by_len)

        try:
            all_records = read_json(args.input_file)
        except:
            all_records = read_jsonl(args.input_file)

        kept_indices = [entry[0] for entry in texts]
        sorted_indices = sorted(kept_indices)
        filtered_records = [all_records[i] for i in sorted_indices]

        write_jsonl(filtered_records, args.filtered_data_file)
        print(f"Rank 0: Wrote {len(filtered_records)} filtered samples to {args.filtered_data_file}")

    # Broadcasting data from rank 0 to other processes 
    # To guarantee that all processess work with the same data
    data_to_broadcast = [texts]
    dist.broadcast_object_list(data_to_broadcast, src=0)
    texts = data_to_broadcast[0]
    
    dataset = TextDataset(texts)
    # DistributedSampler automatically splits data between processes
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # collate_fn to process dict batch
    def collate_fn(batch):
        return {
            "texts": [item["text"] for item in batch],
            "indices": [item["original_index"] for item in batch]
        }

    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn)
    local_results = [] # Current process results
    
    progress_bar = tqdm(dataloader, desc=f"Processing on Rank {rank}", disable=(rank != 0))

    with torch.no_grad():
        for batch in progress_bar:
            inputs = tokenizer(
                batch["texts"],
                return_tensors='pt',
                padding=True,
                max_length=args.max_length
            ).to(device)
            
            outputs = model(**inputs)
            logits = outputs.logits  # [batch_size, sequence_length, vocab_size]

            if args.per_token:
                # Top-k for each token and filter out padding tokens
                topk = torch.topk(logits, k=args.topk, dim=-1)
                indices = topk.indices.cpu().numpy().astype(np.int32).tolist()
                values = topk.values.cpu().float().numpy().astype(np.float16).tolist()
            else:
                # Top-k only for the last token
                last_logits = logits[:, -1, :]
                topk = torch.topk(last_logits, k=args.topk, dim=-1)
                indices = topk.indices.cpu().numpy().astype(np.int32).tolist()
                values = topk.values.cpu().float().numpy().astype(np.float16).tolist()
            
            # Save results along with originla indices
            for i in range(len(batch["indices"])):
                local_results.append({
                 "original_index": batch["indices"][i],
                 "top_k_indices": indices[i],
                 "top_k_values": values[i],
                 #"mask": shifted_attn[i].cpu().tolist(),
                 #"next_ids": next_ids[i].cpu().tolist(),
             })
    print(f"Rank {rank}: Finished processing. Waiting to gather results...")
    
    # Gathering data from all processes to rank 0
    cpu_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")

    chunk_size = max(1, 10000 // args.topk)
    gathered = [] if rank == 0 else None
    num_chunks = (len(local_results) + chunk_size - 1) // chunk_size
    for i in range(num_chunks):
        chunk = local_results[i*chunk_size : (i+1)*chunk_size]
        chunk_lists = [None] * world_size if rank == 0 else None
        dist.gather_object(
            chunk,
            chunk_lists,
            dst=0,
            group=cpu_group
        )
        if rank == 0:
            for sub in chunk_lists:
                gathered.extend(sub)

    dist.barrier(group=cpu_group)
    cleanup_distributed()

    
    if rank == 0:
        print("Rank 0: Gathering and saving results...")
        final_results_flat = gathered
        final_results_flat.sort(key=lambda x: x["original_index"])
        seen = set()
        filtered_results = []
        for res in tqdm(final_results_flat, desc="Filtering duplicates"):
            idx = res["original_index"]
            if idx not in seen:
                seen.add(idx)
                filtered_results.append(res)

        print(len(final_results_flat))
        print(len(filtered_results))

        out_dir = os.path.dirname(args.output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        
        # Saving topk and idx to JSON file (non-efficient)
        #with open(args.output_file, 'w', encoding='utf-8') as f:
        #    for res in filtered_results:
        #        record = {
        #            "idx": res["original_index"],
        #            "top_k_indices": res["top_k_indices"],
        #            "top_k_values": res["top_k_values"],
        #            #"mask": res['mask'],
        #            #"next_ids": res['next_ids'],
        #        }
        #        f.write(json.dumps(record, ensure_ascii=False) + "\n")

        parquet_dir = os.path.dirname(args.output_file)
        if parquet_dir:
            os.makedirs(parquet_dir, exist_ok=True)

               
        del gathered
        del final_results_flat

        if args.per_token:
            indices_type = pa.list_(pa.list_(pa.int32()))
            values_type = pa.list_(pa.list_(pa.float32()))
        else:
            indices_type = pa.list_(pa.int32())
            values_type = pa.list_(pa.float32())

        schema = pa.schema([
            pa.field("idx", pa.int64()),
            pa.field("topk_indices", indices_type),
            pa.field("topk_values", values_type),
        ])

        batch_size = 5000
        with pq.ParquetWriter(args.output_file, schema, compression="gzip") as writer:
            for start_idx in tqdm(range(0, len(filtered_results), batch_size), desc="Writing parquet (streaming)"):
                batch = filtered_results[start_idx:start_idx + batch_size]

                idx_array = pa.array([int(res["original_index"]) for res in batch], type=pa.int64())
                if args.per_token:
                    indices_list = pa.array(
                        [np.asarray(res["top_k_indices"], dtype=np.int32).tolist() for res in batch],
                        type=indices_type,
                    )
                    values_list = pa.array(
                        [np.asarray(res["top_k_values"], dtype=np.float32).tolist() for res in batch],
                        type=values_type,
                    )
                else:
                    indices_list = pa.array(
                        [np.asarray(res["top_k_indices"], dtype=np.int32).tolist() for res in batch],
                        type=indices_type,
                    )
                    values_list = pa.array(
                        [np.asarray(res["top_k_values"], dtype=np.float32).tolist() for res in batch],
                        type=values_type,
                    )
                table = pa.Table.from_arrays([idx_array, indices_list, values_list],
                                             names=["idx", "topk_indices", "topk_values"])
                writer.write_table(table)
        
        print(f"Successfully processed {len(filtered_results)} texts and saved to {args.output_file}")

    if rank == 0:
        formatted = time.strftime('%H:%M:%S', time.gmtime(time.time() - start))
        print(f"Total execution time: {formatted}")

if __name__ == "__main__":
    main()
