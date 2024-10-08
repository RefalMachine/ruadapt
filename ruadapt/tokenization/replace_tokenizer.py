from utils import get_tokenizer_properties, convert_token_to_string_universal, convert_token_universal
from torch import nn
from tqdm import tqdm
import torch
import codecs
import json


def reinit_embeddings_with_head_universal(model, tokenizer_src, tokenizer_dst, mode='random', lm_head_init='tie', add_special_tokens_src=True, mean_cutoff=None):
    assert lm_head_init in ['tie', 'hm']
    assert mode in ['random', 'mean']
    assert model.lm_head.bias is None

    if mean_cutoff is None:
        mean_cutoff = model.model.embed_tokens.weight.shape[0]

    assert model.model.embed_tokens.weight.shape[0] >= mean_cutoff
    assert model.lm_head.weight.shape[0]            >= mean_cutoff

    tokenizer_src_prop = get_tokenizer_properties(tokenizer_src)
    tokenizer_dst_prop = get_tokenizer_properties(tokenizer_dst)
    if add_special_tokens_src:
        # situation where additional_special_tokens already in src and dst tokenizers not tested
        assert 'additional_special_tokens' not in tokenizer_dst.special_tokens_map or 'additional_special_tokens' not in tokenizer_src.special_tokens_map or tokenizer_src.special_tokens_map['additional_special_tokens'] == tokenizer_dst.special_tokens_map['additional_special_tokens']
        special_tokens_map = {key: val for key, val in tokenizer_src.special_tokens_map.items() if key != 'additional_special_tokens'}
        special_tokens_map['additional_special_tokens'] = list(tokenizer_src.added_tokens_decoder.values())
        tokenizer_dst.add_special_tokens(special_tokens_map)
    vocab_size = len(tokenizer_dst.get_vocab())
    print(f'New vocab size (maybe without new special tokens): {vocab_size}')
    model.config.vocab_size = vocab_size
    torch_dtype = model.model.embed_tokens.weight.dtype
    print(f'dtype = {torch_dtype}')

    embeddings_src = model.model.embed_tokens.weight.data.clone().to(torch.float32)
    lm_head_src = model.lm_head.weight.data.clone().to(torch.float32)
    
    model.model.embed_tokens = nn.Embedding(model.config.vocab_size, model.config.hidden_size, dtype=torch_dtype)
    model.lm_head = nn.Linear(model.config.hidden_size, model.config.vocab_size, bias=False, dtype=torch_dtype)

    model.model.embed_tokens.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
    model.lm_head.weight.data.normal_(mean=0.0, std=model.config.initializer_range)

    logs = []
    tokenizer_src_vocab = tokenizer_src.get_vocab()
    if mode == 'mean':
        spec_tokens = set()
        for key in tokenizer_dst.special_tokens_map:
            if key == 'additional_special_tokens':
                spec_tokens.update(set(tokenizer_dst.special_tokens_map['additional_special_tokens']))
            else:
                spec_tokens.add(tokenizer_dst.special_tokens_map[key])

        with torch.no_grad():
            input_emb_mean = torch.mean(embeddings_src[:mean_cutoff], dim=0).to(torch_dtype)
            output_emb_mean = torch.mean(lm_head_src[:mean_cutoff], dim=0).to(torch_dtype)

            for i in tqdm(range(vocab_size)):
                token = tokenizer_dst._tokenizer.id_to_token(i)
                if token in spec_tokens:
                    token_idx = tokenizer_src._tokenizer.token_to_id(token)
                    embed_tokens_ids = [token_idx]
                    if token_idx is None:
                        embed_tokens_ids = None
                    token_str = token
                else:
                    token_str = convert_token_to_string_universal(token, tokenizer_dst, tokenizer_src_vocab, tokenizer_dst_prop)
                    embed_tokens_ids = convert_token_universal(token_str, tokenizer_src, tokenizer_src_vocab, tokenizer_src_prop)

                logs.append({'token_id': i, 'token_repr': token, 'token_str': token_str, 'tokens_src': embed_tokens_ids})

                if embed_tokens_ids is None:
                    input_emb_vec = input_emb_mean
                else:
                    input_emb_vec = embeddings_src[embed_tokens_ids].mean(axis=0).to(torch_dtype)

                if input_emb_vec.norm() < 1e-12:
                    logs[-1]['input_emb_vec_mean'] = True
                    input_emb_vec = input_emb_mean
                model.model.embed_tokens.weight.data[i].copy_(input_emb_vec)

                if lm_head_init == 'hm':
                    if embed_tokens_ids is None:
                        output_emb_vec = output_emb_mean
                    else:
                        output_emb_vec = lm_head_src[embed_tokens_ids].mean(axis=0).to(torch_dtype)
                elif lm_head_init == 'tie':
                    output_emb_vec = input_emb_vec
                
                if output_emb_vec.norm() < 1e-12:
                    logs[-1]['output_emb_vec_mean'] = True
                    output_emb_vec = output_emb_mean
                model.lm_head.weight.data[i].copy_(output_emb_vec)

    elif mode == 'random':
        pass
    else:
        raise Exception('NotImplemented')
    return logs