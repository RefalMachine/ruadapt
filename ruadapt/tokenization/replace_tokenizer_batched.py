import os
import sys
import torch
from torch import nn
from tqdm import tqdm
import json
import numpy as np
import sys
from .utils import get_tokenizer_properties, convert_token_to_string_universal, convert_token_universal

# ==========================================
# MLP Head Architecture 
# ==========================================
import math

class LayerAttentionHead(nn.Module):
    def __init__(self, hidden_size: int, num_stored_layers: int, pooling: str = "last"):
        super().__init__()
        self.pooling = pooling
        self.hidden_size = hidden_size
        self.layer_weights = nn.Parameter(torch.ones(num_stored_layers, dtype=torch.float32))

        if pooling == "attention":
            self.token_attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.Tanh(),
                nn.Linear(hidden_size // 4, 1),
            )
        elif pooling == "global_query":
            self.global_query = nn.Parameter(
                torch.randn(hidden_size) / math.sqrt(hidden_size)
            )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, stacked: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stacked:        [B, L, S, H]  float16 on device
            attention_mask: [B, S]        bool on device
        Returns:
            [B, H]  bfloat16
        """
        # --- Layer fusion in float32 ---
        weights = torch.softmax(self.layer_weights, dim=0)  # [L] float32
        fused = (stacked.float() * weights.view(1, -1, 1, 1)).sum(dim=1)  # [B, S, H] float32

        # --- Sequence pooling ---
        if self.pooling == "last":
            seq_lengths = attention_mask.sum(dim=1).long() - 1          # [B]
            pooled = fused[torch.arange(fused.size(0), device=fused.device), seq_lengths]

        elif self.pooling == "mean":
            mask_f = attention_mask.unsqueeze(-1).float()                # [B, S, 1]
            pooled = (fused * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-9)

        elif self.pooling == "attention":
            attn_in = fused
            scores = self.token_attention(attn_in).squeeze(-1)           # [B, S]
            scores = scores.masked_fill(~attention_mask, float("-inf"))
            attn_w = torch.softmax(scores, dim=1).unsqueeze(-1)
            pooled = (attn_in * attn_w).sum(dim=1)

        elif self.pooling == "global_query":
            scale = math.sqrt(self.hidden_size)
            scores = torch.einsum("bsh,h->bs", fused, self.global_query.float()) / scale
            scores = scores.masked_fill(~attention_mask, float("-inf"))
            attn_w = torch.softmax(scores, dim=1).unsqueeze(-1)         # [B, S, 1]
            pooled = (fused * attn_w).sum(dim=1)                        # [B, H] float32

        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # --- MLP projection (in float32) ---
        return self.mlp(pooled.float()).to(torch.bfloat16)              # [B, H] bfloat16


# ==========================================
# Helpers
# ==========================================
def get_embed_layer(model):
    if hasattr(model, 'get_input_embeddings') and model.get_input_embeddings() is not None:
        return model, model.get_input_embeddings()
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        return model.model.language_model, model.model.language_model.embed_tokens
    if hasattr(model, 'language_model'):
        submodel = model.language_model.model if hasattr(model.language_model, 'model') else model.language_model
        return submodel, submodel.embed_tokens
    return model.model, model.model.embed_tokens

def get_lm_head(model):
    if hasattr(model, 'get_output_embeddings') and model.get_output_embeddings() is not None:
        return model, model.get_output_embeddings()
    if hasattr(model, 'lm_head'):
        return model, model.lm_head
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'lm_head'):
        return model.language_model, model.language_model.lm_head
    return model, getattr(model, 'lm_head', None)

def get_weights(tokens, mult=1.0, is_space_mask=None, space_penalty=0.1):
    raw_weights = [np.exp(-mult*i) for i in range(len(tokens))]
    if is_space_mask is not None and len(tokens) > 1:
        for i in range(len(tokens)):
            if is_space_mask[i]:
                raw_weights[i] *= space_penalty
    norm = sum(raw_weights)
    if norm == 0: norm = 1e-12
    return torch.tensor([w / norm for w in raw_weights])

def weight_average(tokens_tensor, mult=1.0, is_space_mask=None, space_penalty=0.1):
    weights = get_weights(tokens_tensor, mult, is_space_mask, space_penalty).to(tokens_tensor.device)
    return (tokens_tensor * weights.unsqueeze(-1)).sum(dim=0)

def predict_vec(input_ids, model, mlp_head):
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(input_ids=torch.tensor([input_ids], device=model.device), output_hidden_states=True)
        
        # Ensure we only pass the exact number of layers the MLP head expects (from the top)
        hidden_states = torch.stack(list(outputs.hidden_states)[-25:], dim=1)
        attention_mask = torch.ones((1, len(input_ids)), dtype=torch.bool, device=model.device)
        pred_embeds = mlp_head(hidden_states, attention_mask).to(model.device)[0]
    return pred_embeds
# ==========================================
# Core Batched Logic
# ==========================================
def reinit_embeddings_with_head_universal_batched(
    model, tokenizer_src, tokenizer_dst, 
    mode='mean', lm_head_init='tie', add_special_tokens_src=True, 
    mean_cutoff=None, mult=1.0, 
    head_path=None, pooling='attention', batch_size=256
):
    assert lm_head_init in ['tie', 'hm']
    assert mode in ['random', 'mean', 'wmean', 'mlp']

    model.eval()

    embed_parent, embed_tokens = get_embed_layer(model)
    head_parent, lm_head = get_lm_head(model)

    if mean_cutoff is None:
        mean_cutoff = embed_tokens.weight.shape[0]

    # Handle Special Tokens Mapping
    tokenizer_src_prop = get_tokenizer_properties(tokenizer_src)
    tokenizer_dst_prop = get_tokenizer_properties(tokenizer_dst)
    if add_special_tokens_src:
        special_tokens_map = {key: val for key, val in tokenizer_src.special_tokens_map.items() if key != 'additional_special_tokens'}
        if 'additional_special_tokens' in tokenizer_src.special_tokens_map:
            special_tokens_map['additional_special_tokens'] = list(tokenizer_src.added_tokens_decoder.values())
        tokenizer_dst.add_special_tokens(special_tokens_map)
        
    new_vocab_size = len(tokenizer_dst.get_vocab())
    print(f'New vocab size: {new_vocab_size}')
    torch_dtype = embed_tokens.weight.dtype

    # Clone source weights
    embeddings_src = embed_tokens.weight.data.clone().float()
    lm_head_src = lm_head.weight.data.clone().float() if lm_head is not None else None
    
    hidden_size = getattr(model.config, 'hidden_size', embed_tokens.weight.shape[1])
    is_tied = getattr(model.config, 'tie_word_embeddings', False)

    # Calculate fallback means
    input_emb_mean = torch.mean(embeddings_src[:mean_cutoff], dim=0).to(torch_dtype).to(model.device)
    output_emb_mean = None
    if not is_tied and lm_head_src is not None:
        output_emb_mean = torch.mean(lm_head_src[:mean_cutoff], dim=0).to(torch_dtype).to(model.device)

    # Pre-allocate offline tensors for the new architecture
    new_embed_weight = torch.zeros((new_vocab_size, hidden_size), dtype=torch_dtype, device=model.device)
    new_head_weight = torch.zeros((new_vocab_size, hidden_size), dtype=torch_dtype, device=model.device) if not is_tied else None

    # Initialize with normal distribution
    init_range = getattr(model.config, 'initializer_range', 0.02)
    new_embed_weight.normal_(mean=0.0, std=init_range)
    if not is_tied:
        new_head_weight.normal_(mean=0.0, std=init_range)

    # Copy over original vocabulary
    copy_limit = min(mean_cutoff, new_vocab_size)
    new_embed_weight[:copy_limit] = embed_tokens.weight.data[:copy_limit]
    if not is_tied and lm_head is not None:
        new_head_weight[:copy_limit] = lm_head.weight.data[:copy_limit]

    # Load MLP Head if needed
    mlp_head = None
    if mode == 'mlp':
        print(f"Loading MLP Head from {head_path} (pooling={pooling})")
        state_dict = torch.load(head_path, map_location=model.device)
        if all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
            
        # Dynamically determine expected number of layers from the checkpoint
        num_layers = state_dict['layer_weights'].shape[0]
        print(f"Detected {num_layers} layers in saved MLP head.")
        
        mlp_head = LayerAttentionHead(hidden_size, num_layers, pooling).to(model.device)
        mlp_head.load_state_dict(state_dict)
        mlp_head.eval()

    # Step 1: Analyze new tokens and group tasks
    tasks = []
    logs = []
    spec_tokens = set(tokenizer_dst.special_tokens_map.values())
    if 'additional_special_tokens' in tokenizer_dst.special_tokens_map:
        spec_tokens.update(tokenizer_dst.special_tokens_map['additional_special_tokens'])

    tokenizer_src_vocab = tokenizer_src.get_vocab()
    
    print("Analyzing vocabulary mapping...")
    for i in tqdm(range(copy_limit, new_vocab_size)):
        token = tokenizer_dst._tokenizer.id_to_token(i)
        token_decoded = tokenizer_dst.convert_tokens_to_string([token])
        if token is None:
            logs.append({'token_id': i, 'token_repr': None, 'tokens_src': None, 'input_emb_vec_mean': True})
            new_embed_weight[i] = input_emb_mean
            if not is_tied:
                new_head_weight[i] = output_emb_mean if lm_head_init == 'hm' else input_emb_mean
            continue

        if token in spec_tokens:
            token_idx = tokenizer_src._tokenizer.token_to_id(token)
            embed_tokens_ids = [token_idx] if token_idx is not None else None
            token_str = token
        else:
            token_str = convert_token_to_string_universal(token, tokenizer_dst, tokenizer_src_vocab, tokenizer_dst_prop)
            embed_tokens_ids = convert_token_universal(token_str, tokenizer_src, tokenizer_src_vocab, tokenizer_src_prop)

        is_space_mask = None
        if embed_tokens_ids is not None:
            is_space_mask = []
            for tid in embed_tokens_ids:
                t_str = tokenizer_src._tokenizer.id_to_token(tid)
                if t_str is not None and t_str.replace(' ', '').replace('Ġ', '').replace('▁', '') == '':
                    is_space_mask.append(True)
                else:
                    is_space_mask.append(False)

        logs.append({'token_id': i, 'token_repr': token, 'token_str': token_str, 'tokens_src': embed_tokens_ids})
        
        if embed_tokens_ids is None:
            new_embed_weight[i] = input_emb_mean
            if not is_tied:
                new_head_weight[i] = output_emb_mean if lm_head_init == 'hm' else input_emb_mean
        else:
            
            tasks.append({
                'idx': i,
                'token': token,
                'token_decoded': token_decoded,
                'src_ids': embed_tokens_ids,
                'space_mask': is_space_mask
            })
            if token_decoded == ' историческая':
                print(tasks[-1])
                print(predict_vec(embed_tokens_ids, model, mlp_head))

    # Step 2: Batched Generation
    if len(tasks) > 0 and mode != 'random':
        print(f"Processing {len(tasks)} tokens in {mode} mode (batches of {batch_size})...")
        for b_start in tqdm(range(0, len(tasks), batch_size)):
            batch_tasks = tasks[b_start : b_start + batch_size]
            #if b_start % 100 == 0:
            #    print(batch_tasks)
            if mode in ['mean', 'wmean']:
                for task in batch_tasks:
                    idx = task['idx']
                    src_ids = task['src_ids']
                    s_mask = task['space_mask']
                    
                    m = 0.0 if mode == 'mean' else mult
                    src_tensors = embeddings_src[src_ids].to(model.device)
                    new_embed_weight[idx] = weight_average(src_tensors, mult=m, is_space_mask=s_mask, space_penalty=0.1).to(torch_dtype)
                    
                    if not is_tied:
                        if lm_head_init == 'hm':
                            head_src_tensors = lm_head_src[src_ids].to(model.device)
                            new_head_weight[idx] = weight_average(head_src_tensors, mult=m, is_space_mask=s_mask, space_penalty=0.1).to(torch_dtype)
                        else:
                            new_head_weight[idx] = new_embed_weight[idx]

            elif mode == 'mlp':
                max_len = max(len(t['src_ids']) for t in batch_tasks)
                input_ids = []
                attention_mask = []
                
                for task in batch_tasks:
                    ids = torch.tensor(task['src_ids'], dtype=torch.long)
                    pad_len = max_len - len(ids)
                    pad_id = getattr(tokenizer_src, 'pad_token_id', None)
                    if pad_id is None:
                        pad_id = getattr(tokenizer_src, 'eos_token_id', 0)
                    if pad_id is None:
                        pad_id = 0
                    input_ids.append(torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)]))
                    attention_mask.append(torch.cat([torch.ones(len(ids), dtype=torch.bool), torch.zeros(pad_len, dtype=torch.bool)]))
                
                #if b_start % 100 == 0:
                #    print(input_ids)
                #    print(attention_mask)
                input_ids = torch.stack(input_ids).to(model.device)
                attention_mask = torch.stack(attention_mask).to(model.device)
                
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    
                    # Ensure we only pass the exact number of layers the MLP head expects (from the top)
                    hidden_states = torch.stack(list(outputs.hidden_states)[-num_layers:], dim=1)
                    pred_embeds = mlp_head(hidden_states, attention_mask).to(torch_dtype)
                
                for i, task in enumerate(batch_tasks):
                    idx = task['idx']
                    if idx == 260577:
                        print(pred_embeds[i])
                    if len(task['src_ids']) == 1:
                        # Single-token (including special tokens) perfectly mapped, bypass MLP to avoid noise
                        src_id = task['src_ids'][0]
                        new_embed_weight[idx] = embed_tokens.weight.data[src_id].to(torch_dtype)
                        if not is_tied:
                            new_head_weight[idx] = lm_head_src[src_id].to(torch_dtype)
                    else:
                        new_embed_weight[idx] = pred_embeds[i]
                        
                        if not is_tied:
                            if lm_head_init == 'hm':
                                m = mult
                                head_src_tensors = lm_head_src[task['src_ids']].to(model.device)
                                new_head_weight[idx] = weight_average(head_src_tensors, mult=m, is_space_mask=task['space_mask'], space_penalty=0.1).to(torch_dtype)
                            else:
                                new_head_weight[idx] = pred_embeds[i]

    # Step 3: Surgical Replacement
    print("Surgical injection of new vocabulary weights...")
    print(new_embed_weight[260577])
    model.config.vocab_size = new_vocab_size
    
    new_embed = torch.nn.Embedding(new_vocab_size, hidden_size, dtype=torch_dtype, device=model.device)
    new_head = torch.nn.Linear(hidden_size, new_vocab_size, bias=False, dtype=torch_dtype, device=model.device)

    # Assign gathered weights
    new_embed.weight.data.copy_(new_embed_weight)
    if not is_tied and lm_head is not None:
        new_head.weight.data.copy_(new_head_weight)

    # Hard replacement mirroring replace_tokenizer.py
    if hasattr(model, 'set_input_embeddings'):
        model.set_input_embeddings(new_embed)
        embed_tokens = model.get_input_embeddings()
    else:
        embed_parent.embed_tokens = new_embed
        embed_tokens = new_embed

    if hasattr(model, 'set_output_embeddings'):
        model.set_output_embeddings(new_head)
        lm_head = model.get_output_embeddings()
    else:
        head_parent.lm_head = new_head
        lm_head = new_head

    if is_tied:
        print("Detected tied word embeddings. Hard-tying lm_head to embed_tokens.")
        lm_head.weight = embed_tokens.weight

    return logs
