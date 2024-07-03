from torch import nn
from tqdm import tqdm
import torch
import json

def load_occs(path):
    with open(path, encoding='utf-8') as f:
        return {int(k) : {int(kk):int(vv) for kk,vv in v.items()}for k,v in json.load(f).items()}
 
def convert_tok_map_to_probs(tok_map):
    for k,v in tok_map.items():
        norm=sum(v.values())
        for vv in v.keys():
            tok_map[k][vv]/=norm
 
MODEL_ALIAS = "llama-3"
            
occ_tok_map=load_occs(f"vocab_init_data/{MODEL_ALIAS}_cooccurrence.json")
convert_tok_map_to_probs(occ_tok_map)
 
#def special_encode(token_str, tokenizer):
    #if "llama" in tokenizer.name_or_path.lower():
    #    token_str = token_str.replace('▁', ' ')
    #shift = len(tokenizer.encode('家', add_special_tokens=False))
    #tokens = tokenizer.encode('家' + token_str, add_special_tokens=False)
    #return tokens[shift:]
#    return tokenizer.encode(token_str, add_special_tokens=False)
    
def get_agg_vec(new_token, new_token_id, old_tokenizer, embeddings, agg_mode='mean'):
    if agg_mode == 'occ' and new_token_id in occ_tok_map:
        subtokens, probs = list(zip(*sorted(occ_tok_map[new_token_id].items())))
        subtokens = list(subtokens)
        # print(subtokens)
    else:
        subtokens = special_encode(new_token, old_tokenizer)
    vector = embeddings[subtokens]
    if len(subtokens)>1:
        if agg_mode=='mean':
            vector=vector.mean(axis=0)
        elif agg_mode=='sum':
            vector=vector.sum(axis=0)
        elif agg_mode=='occ':
            if new_token_id in occ_tok_map:
                probs=torch.tensor(probs, dtype=vector.dtype).reshape(-1,1)
                probs = probs.to(embeddings.device)
                vector=(probs * vector).sum(axis=0)
            else:
                print(new_token_id)
                vector=vector.mean(axis=0)
        else:
            raise ValueError
    else:
        vector=vector.reshape(-1)
    return vector

def convert_ascii_hex(token):
    return int(token[-2], 16) + 16 * int(token[-3], 16)

def if_hex(token):
    return token.startswith('<0x') and token.endswith('>')

def simple_encode(token_str, tokenizer):
    #if "llama" in tokenizer.name_or_path.lower():
    #    token_str = token_str.replace('▁', ' ')
    #shift = len(tokenizer.encode('家', add_special_tokens=False))
    #tokens = tokenizer.encode('家' + token_str, add_special_tokens=False)
    #return tokens[shift:]
    return tokenizer.encode(token_str, add_special_tokens=False)

def special_encode(token_str, tokenizer):
    shift = len(tokenizer.encode('家', add_special_tokens=False))
    tokens = tokenizer.encode('家' + token_str, add_special_tokens=False)
    if token_str == '\n':
        print(tokens)
        print(tokens[shift:])
    return tokens[shift:]

def get_mean_vec(token, tokenizer, embeddings, encode_func):
    tokens = encode_func(token, tokenizer)
    if token == 'онный':
        print(tokens)
    vector = embeddings[tokens].mean(axis=0)
    return vector

def get_sum_vec(token, tokenizer, embeddings):
    tokens = special_encode(token, tokenizer)
    if token == 'онный':
        print(tokens)
    vector = embeddings[tokens].sum(axis=0)
    return vector

def reinit_embeddings_with_head_llama3_extended(model, tokenizer_old, tokenizer_new, mode='random', lm_head_init='tie'):
    assert lm_head_init in ['tie', 'hm']
    assert mode in ['random', 'mean', 'occ']
    assert model.lm_head.bias is None

    #tokenizer_new.add_special_tokens({'bos_token': tokenizer_old.added_tokens_decoder[128000], 'eos_token': tokenizer_old.added_tokens_decoder[128001], 'additional_special_tokens': list(tokenizer_old.added_tokens_decoder.values())})
    
    vocab_size = len(tokenizer_new.get_vocab())
    #tokenizer_new.vocab_size = vocab_size
    model.config.vocab_size = vocab_size

    embeddings_old = model.model.embed_tokens.weight.data.clone()
    lm_head_old = model.lm_head.weight.data.clone()
    
    model.model.embed_tokens = nn.Embedding(model.config.vocab_size, model.config.hidden_size, dtype=embeddings_old.dtype)
    model.lm_head = nn.Linear(model.config.hidden_size, model.config.vocab_size, bias=False, dtype=embeddings_old.dtype)

    if mode == 'random':
        model.model.embed_tokens.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        model.lm_head.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
    elif mode == 'mean' or mode == 'occ':
        model.model.embed_tokens.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        model.lm_head.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        '''
        spec_tokens = set(tokenizer_new.special_tokens_map['additional_special_tokens']) if 'additional_special_tokens' in tokenizer_new.special_tokens_map else set()
        if 'eos_token' in tokenizer_new.special_tokens_map:
            spec_tokens.add(tokenizer_new.special_tokens_map['eos_token'])
        if 'bos_token' in tokenizer_new.special_tokens_map:
            spec_tokens.add(tokenizer_new.special_tokens_map['bos_token'])
        #if 'unk_token' in tokenizer_new.special_tokens_map:
        #    spec_tokens.add(tokenizer_new.special_tokens_map['unk_token'])
        if 'pad_token' in tokenizer_new.special_tokens_map:
            spec_tokens.add(tokenizer_new.special_tokens_map['pad_token'])
        '''
        zero_tokens_ids = set()
        for i in range(embeddings_old.shape[0]):
            if embeddings_old[i].norm() < 1e-12:
                print(i)
                zero_tokens_ids.add(i)

        print(len(zero_tokens_ids))
        with torch.no_grad():
            for i in tqdm(range(vocab_size)):
                if i in zero_tokens_ids:
                    model.model.embed_tokens.weight.data[i].copy_(embeddings_old[128009])
                    model.lm_head.weight.data[i].copy_(lm_head_old[128009])
                    continue
                token = tokenizer_new._tokenizer.id_to_token(i)
                token = tokenizer_new.convert_tokens_to_string([token])
                #vec = get_agg_vec(token, i, tokenizer_old, embeddings_old, mode)#get_mean_vec(token, tokenizer_old, embeddings_old)
                vec = get_mean_vec(token, tokenizer_old, embeddings_old, simple_encode)
                if vec.norm() < 1e-9:
                    print('skip ' + token)
                else:
                    model.model.embed_tokens.weight.data[i].copy_(vec)

                if lm_head_init == 'hm': # lm head mean
                    #vec = get_agg_vec(token, i, tokenizer_old, lm_head_old, mode)#get_mean_vec(token, tokenizer_old, lm_head_old)
                    vec = get_mean_vec(token, tokenizer_old, lm_head_old, simple_encode)
                    if vec.norm() < 1e-9:
                        print('skip ' + token)
                    else:
                        model.lm_head.weight.data[i].copy_(vec)
                elif lm_head_init == 'tie':
                    model.lm_head.weight.data[i].copy_(vec)
                    
                #if lm_head_init == 'tie':
                #    model._tie_or_clone_weights(model.get_output_embeddings(), model.get_input_embeddings())
    else:
        raise Exception('NotImplemented')

    print(model.model.embed_tokens.weight.data[0])
    print(model.lm_head.weight.data[0])

def get_hex_vec(token, tokenizer_old, embeddings):
    info = []
    info.append([token, tokenizer_old.vocab.get(token, -1)])
    if info[-1][1] != -1:
        return embeddings[info[-1][1]]
    token = chr(convert_ascii_hex(token))
    info.append([token, tokenizer_old.vocab.get(token, -1)])
    if info[-1][1] != -1:
        return embeddings[info[-1][1]]

    return get_mean_vec(token, tokenizer_old, embeddings, simple_encode)

def reinit_embeddings_with_head_llama3(model, tokenizer_old, tokenizer_new, mode='random', lm_head_init='tie'):
    assert lm_head_init in ['tie', 'hm']
    assert mode in ['random', 'mean', 'occ']
    assert model.lm_head.bias is None

    tokenizer_new.add_special_tokens({'bos_token': tokenizer_old.added_tokens_decoder[128000], 'eos_token': tokenizer_old.added_tokens_decoder[128001], 'additional_special_tokens': list(tokenizer_old.added_tokens_decoder.values())})
    
    vocab_size = len(tokenizer_new.get_vocab())
    #tokenizer_new.vocab_size = vocab_size
    model.config.vocab_size = vocab_size

    embeddings_old = model.model.embed_tokens.weight.data.clone()
    lm_head_old = model.lm_head.weight.data.clone()
    
    model.model.embed_tokens = nn.Embedding(model.config.vocab_size, model.config.hidden_size, dtype=embeddings_old.dtype)
    model.lm_head = nn.Linear(model.config.hidden_size, model.config.vocab_size, bias=False, dtype=embeddings_old.dtype)

    if mode == 'random':
        model.model.embed_tokens.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        model.lm_head.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
    elif mode == 'mean' or mode == 'occ':
        model.model.embed_tokens.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        model.lm_head.weight.data.normal_(mean=0.0, std=model.config.initializer_range)

        spec_tokens = set(tokenizer_new.special_tokens_map['additional_special_tokens']) if 'additional_special_tokens' in tokenizer_new.special_tokens_map else set()
        if 'eos_token' in tokenizer_new.special_tokens_map:
            spec_tokens.add(tokenizer_new.special_tokens_map['eos_token'])
        if 'bos_token' in tokenizer_new.special_tokens_map:
            spec_tokens.add(tokenizer_new.special_tokens_map['bos_token'])
        #if 'unk_token' in tokenizer_new.special_tokens_map:
        #    spec_tokens.add(tokenizer_new.special_tokens_map['unk_token'])
        if 'pad_token' in tokenizer_new.special_tokens_map:
            spec_tokens.add(tokenizer_new.special_tokens_map['pad_token'])

        with torch.no_grad():
            for i in tqdm(range(vocab_size)):
                token = tokenizer_new._tokenizer.id_to_token(i)
                if token in spec_tokens:
                    vec = embeddings_old[tokenizer_old._tokenizer.token_to_id(token)]
                elif if_hex(token):
                    vec = get_hex_vec(token, tokenizer_old, embeddings_old)
                else:
                    token = token.replace('▁', ' ')
                    vec = get_mean_vec(token, tokenizer_old, embeddings_old, simple_encode)#get_agg_vec(token, i, tokenizer_old, embeddings_old, mode)#get_mean_vec(token, tokenizer_old, embeddings_old)

                model.model.embed_tokens.weight.data[i].copy_(vec)

                if lm_head_init == 'hm': # lm head mean
                    if token in spec_tokens:
                        vec = lm_head_old[tokenizer_old._tokenizer.token_to_id(token)]
                    elif if_hex(token):
                        vec = get_hex_vec(token, tokenizer_old, lm_head_old)
                    else:
                        vec = get_mean_vec(token, tokenizer_old, lm_head_old, simple_encode)#get_agg_vec(token, i, tokenizer_old, lm_head_old, mode)#get_mean_vec(token, tokenizer_old, lm_head_old)

                    model.lm_head.weight.data[i].copy_(vec)
                elif lm_head_init == 'tie':
                    model.lm_head.weight.data[i].copy_(vec)
                    
                #if lm_head_init == 'tie':
                #    model._tie_or_clone_weights(model.get_output_embeddings(), model.get_input_embeddings())
    else:
        raise Exception('NotImplemented')

    #print(model.model.embed_tokens.weight.data[0])
    #print(model.lm_head.weight.data[0])

def reinit_embeddings_with_head_llama(model, tokenizer_old, tokenizer_new, mode='random', lm_head_init='tie'):
    assert lm_head_init in ['tie', 'hm']
    assert mode in ['random', 'mean']
    assert model.lm_head.bias is None

    vocab_size = len(tokenizer_new.get_vocab())
    model.config.vocab_size = vocab_size

    embeddings_old = model.model.embed_tokens.weight.data.clone()
    lm_head_old = model.lm_head.weight.data.clone()
    
    model.model.embed_tokens = nn.Embedding(model.config.vocab_size, model.config.hidden_size, dtype=embeddings_old.dtype)
    model.lm_head = nn.Linear(model.config.hidden_size, model.config.vocab_size, bias=False, dtype=embeddings_old.dtype)

    if mode == 'random':
        model.model.embed_tokens.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        model.lm_head.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
    elif mode == 'mean':
        model.model.embed_tokens.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        model.lm_head.weight.data.normal_(mean=0.0, std=model.config.initializer_range)

        spec_tokens = set(tokenizer_new.special_tokens_map['additional_special_tokens']) if 'additional_special_tokens' in tokenizer_new.special_tokens_map else set()
        if 'eos_token' in tokenizer_new.special_tokens_map:
            spec_tokens.add(tokenizer_new.special_tokens_map['eos_token'])
        if 'bos_token' in tokenizer_new.special_tokens_map:
            spec_tokens.add(tokenizer_new.special_tokens_map['bos_token'])
        if 'unk_token' in tokenizer_new.special_tokens_map:
            spec_tokens.add(tokenizer_new.special_tokens_map['unk_token'])
        if 'pad_token' in tokenizer_new.special_tokens_map:
            spec_tokens.add(tokenizer_new.special_tokens_map['pad_token'])

        with torch.no_grad():
            for i in tqdm(range(vocab_size)):
                token = tokenizer_new._tokenizer.id_to_token(i)
                if token in spec_tokens:
                    continue
                if if_hex(token) and token in tokenizer_old.vocab:
                    idx = tokenizer_old.vocab[token]
                    vec = embeddings_old[idx]
                else:
                    '''
                    if i in [0, 1, 2, 13, 25, 100, 500, 1000, 13629]:
                        print(i)
                        print('|' + token + '|' + str(len(token)))
                        print('|' + tokenizer_new.convert_tokens_to_string([token]) + '|' + str(len(tokenizer_new.convert_tokens_to_string([token]))))
                    '''
                    vec = get_mean_vec(token, tokenizer_old, embeddings_old, special_encode)
                model.model.embed_tokens.weight.data[i].copy_(vec)

                if lm_head_init == 'hm': # lm head mean
                    if if_hex(token) and token in tokenizer_old.vocab:
                        idx = tokenizer_old.vocab[token]
                        vec = lm_head_old[idx]
                    else:
                        vec = get_mean_vec(token, tokenizer_old, lm_head_old,special_encode)
                    model.lm_head.weight.data[i].copy_(vec)
                elif lm_head_init == 'tie':
                    model.lm_head.weight.data[i].copy_(vec)
                    
                #if lm_head_init == 'tie':
                #    model._tie_or_clone_weights(model.get_output_embeddings(), model.get_input_embeddings())
    else:
        raise Exception('NotImplemented')

    print(model.model.embed_tokens.weight.data[0])
    print(model.lm_head.weight.data[0])
    #return model
    
def reinit_embeddings_with_head_llama_w2v(model, tokenizer_old, tokenizer_new, w2v_model):
    assert model.lm_head.bias is None

    vocab_size = len(tokenizer_new.get_vocab())
    model.config.vocab_size = vocab_size

    embeddings_old = model.model.embed_tokens.weight.data.clone()

    model.model.embed_tokens = nn.Embedding(model.config.vocab_size, model.config.hidden_size)
    model.lm_head = nn.Linear(model.config.hidden_size, model.config.vocab_size, bias=False)

    model.model.embed_tokens.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
    model.lm_head.weight.data.normal_(mean=0.0, std=model.config.initializer_range)

    spec_tokens = set(tokenizer_new.special_tokens_map['additional_special_tokens']) if 'additional_special_tokens' in tokenizer_new.special_tokens_map else set()
    if 'eos_token' in tokenizer_new.special_tokens_map:
        spec_tokens.add(tokenizer_new.special_tokens_map['eos_token'])
    if 'bos_token' in tokenizer_new.special_tokens_map:
        spec_tokens.add(tokenizer_new.special_tokens_map['bos_token'])
    if 'unk_token' in tokenizer_new.special_tokens_map:
        spec_tokens.add(tokenizer_new.special_tokens_map['unk_token'])
    if 'pad_token' in tokenizer_new.special_tokens_map:
        spec_tokens.add(tokenizer_new.special_tokens_map['pad_token'])

    with torch.no_grad():
        for i in tqdm(range(vocab_size)):
            token = tokenizer_new._tokenizer.id_to_token(i)
            if token in spec_tokens:
                old_id = tokenizer_old._tokenizer.token_to_id(token)
                model.model.embed_tokens.weight.data[i] = embeddings_old[old_id]
                continue
            if i not in w2v_model:
                continue
            vec = torch.tensor(w2v_model[i], dtype=torch.float16)
            model.model.embed_tokens.weight.data[i] = vec

    model._tie_or_clone_weights(model.get_output_embeddings(), model.get_input_embeddings())
