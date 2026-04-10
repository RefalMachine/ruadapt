from .utils import get_tokenizer_properties, convert_token_to_string_universal, convert_token_universal
from torch import nn
from tqdm import tqdm
import torch
import codecs
import json
import numpy as np

def get_embed_layer(model):
    if hasattr(model, 'get_input_embeddings'):
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
    tl = len(tokens)
    raw_weights = [np.exp(-mult*i) for i in range(len(tokens))]
    
    # Применяем штраф к пробельным токенам, если слово разбилось больше чем на 1 кусок
    if is_space_mask is not None and len(tokens) > 1:
        for i in range(len(tokens)):
            if is_space_mask[i]:
                raw_weights[i] *= space_penalty
                
    norm = sum(raw_weights)
    if norm == 0:
        norm = 1e-12
        
    weights = [w / norm for w in raw_weights]
    return torch.tensor(weights)

def weight_average(tokens, mult=1.0, is_space_mask=None, space_penalty=0.1):
    weights = get_weights(tokens, mult, is_space_mask, space_penalty)
    return torch.stack([tokens[i] * _w for i, _w in enumerate(weights)]).sum(axis=0)

def reinit_embeddings_with_head_universal(model, tokenizer_src, tokenizer_dst, mode='random', lm_head_init='tie', add_special_tokens_src=True, mean_cutoff=None, mult=1.0):
    assert lm_head_init in ['tie', 'hm']
    assert mode in ['random', 'mean', 'wmean']

    embed_parent, embed_tokens = get_embed_layer(model)
    head_parent, lm_head = get_lm_head(model)

    if lm_head is not None:
        assert getattr(lm_head, 'bias', None) is None

    if mean_cutoff is None:
        mean_cutoff = embed_tokens.weight.shape[0]

    assert embed_tokens.weight.shape[0] >= mean_cutoff
    assert lm_head.weight.shape[0] >= mean_cutoff

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
    torch_dtype = embed_tokens.weight.dtype
    print(f'dtype = {torch_dtype}')

    embeddings_src = embed_tokens.weight.data.clone().to(torch.float32)
    lm_head_src = lm_head.weight.data.clone().to(torch.float32)
    
    hidden_size = getattr(model.config, 'hidden_size', None)
    if hidden_size is None and hasattr(model.config, 'text_config'):
        hidden_size = getattr(model.config.text_config, 'hidden_size', None)
    if hidden_size is None:
        hidden_size = embed_tokens.weight.shape[1]

    new_embed = nn.Embedding(model.config.vocab_size, hidden_size, dtype=torch_dtype, device=model.device)
    new_head = nn.Linear(hidden_size, model.config.vocab_size, bias=False, dtype=torch_dtype, device=model.device)

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

    is_tied = getattr(model.config, 'tie_word_embeddings', False)
    if is_tied:
        print("Detected tied word embeddings. Hard-tying lm_head to embed_tokens.")
        lm_head.weight = embed_tokens.weight

    init_range = getattr(model.config, 'initializer_range', None)
    if init_range is None and hasattr(model.config, 'text_config'):
        init_range = getattr(model.config.text_config, 'initializer_range', None)
    if init_range is None:
        init_range = 0.02 # fallback std default

    embed_tokens.weight.data.normal_(mean=0.0, std=init_range)
    if not is_tied and lm_head.weight is not None:
        lm_head.weight.data.normal_(mean=0.0, std=init_range)

    logs = []
    tokenizer_src_vocab = tokenizer_src.get_vocab()
    if mode == 'mean' or mode == 'wmean':
        spec_tokens = set()
        for key in tokenizer_dst.special_tokens_map:
            if key == 'additional_special_tokens':
                spec_tokens.update(set(tokenizer_dst.special_tokens_map['additional_special_tokens']))
            else:
                spec_tokens.add(tokenizer_dst.special_tokens_map[key])

        with torch.no_grad():
            input_emb_mean = torch.mean(embeddings_src[:mean_cutoff].to(torch.float64), dim=0).to(torch_dtype).to(model.device)
            if not is_tied:
                output_emb_mean = torch.mean(lm_head_src[:mean_cutoff].to(torch.float64), dim=0).to(torch_dtype).to(model.device)

            for i in tqdm(range(vocab_size)):
                token = tokenizer_dst._tokenizer.id_to_token(i)
                is_space_mask = None
                
                # Обработка "дыр" в словаре (зарезервированные, но пустые ID)
                if token is None:
                    logs.append({'token_id': i, 'token_repr': None, 'token_str': None, 'tokens_src': None, 'input_emb_vec_mean': True})
                    embed_tokens.weight.data[i].copy_(input_emb_mean)
                    if not is_tied:
                        if lm_head_init == 'hm':
                            lm_head.weight.data[i].copy_(output_emb_mean)
                        elif lm_head_init == 'tie':
                            lm_head.weight.data[i].copy_(input_emb_mean)
                    continue

                if token in spec_tokens:
                    token_idx = tokenizer_src._tokenizer.token_to_id(token)
                    embed_tokens_ids = [token_idx]
                    if token_idx is None:
                        embed_tokens_ids = None
                    token_str = token
                else:
                    token_str = convert_token_to_string_universal(token, tokenizer_dst, tokenizer_src_vocab, tokenizer_dst_prop)
                    embed_tokens_ids = convert_token_universal(token_str, tokenizer_src, tokenizer_src_vocab, tokenizer_src_prop)
                    
                    if embed_tokens_ids is not None:
                        # Строим маску пробельных токенов (если токен состоит только из ' ' или 'Ġ' или '_')
                        is_space_mask = []
                        for tid in embed_tokens_ids:
                            t_str = tokenizer_src._tokenizer.id_to_token(tid)
                            if t_str is not None and t_str.replace(' ', '').replace('Ġ', '').replace(' ', '').replace('▁', '') == '':
                                is_space_mask.append(True)
                            else:
                                is_space_mask.append(False)

                logs.append({'token_id': i, 'token_repr': token, 'token_str': token_str, 'tokens_src': embed_tokens_ids})

                if embed_tokens_ids is None:
                    input_emb_vec = input_emb_mean
                else:
                    if mode == 'mean': 
                        # Для mean используем wmean с mult=0.0 (равные веса), чтобы задействовать штраф за пробелы
                        input_emb_vec = weight_average(embeddings_src[embed_tokens_ids], mult=0.0, is_space_mask=is_space_mask, space_penalty=0.1).to(torch_dtype).to(model.device)
                    elif mode == 'wmean':
                        input_emb_vec = weight_average(embeddings_src[embed_tokens_ids], mult=mult, is_space_mask=is_space_mask, space_penalty=0.1).to(torch_dtype).to(model.device)

                if input_emb_vec.norm() < 1e-12:
                    logs[-1]['input_emb_vec_mean'] = True
                    input_emb_vec = input_emb_mean
                embed_tokens.weight.data[i].copy_(input_emb_vec)

                # Если веса связаны, embed_tokens уже обновил lm_head, пропускаем
                if is_tied:
                    continue

                if lm_head_init == 'hm':
                    if embed_tokens_ids is None:
                        output_emb_vec = output_emb_mean
                    else:
                        if mode == 'mean': 
                            output_emb_vec = weight_average(lm_head_src[embed_tokens_ids], mult=0.0, is_space_mask=is_space_mask, space_penalty=0.1).to(torch_dtype).to(model.device)
                        elif mode == 'wmean':
                            output_emb_vec = weight_average(lm_head_src[embed_tokens_ids], mult=mult, is_space_mask=is_space_mask, space_penalty=0.1).to(torch_dtype).to(model.device)
                elif lm_head_init == 'tie':
                    output_emb_vec = input_emb_vec
                
                if output_emb_vec.norm() < 1e-12:
                    logs[-1]['output_emb_vec_mean'] = True
                    output_emb_vec = output_emb_mean
                lm_head.weight.data[i].copy_(output_emb_vec)

    elif mode == 'random':
        pass
    else:
        raise Exception('NotImplemented')
    return logs