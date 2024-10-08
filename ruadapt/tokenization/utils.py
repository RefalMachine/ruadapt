from torch import nn
from tqdm import tqdm
import torch

def convert_ascii_hex(token):
    return int(token[-2], 16) + 16 * int(token[-3], 16)

def if_hex(token):
    return token.startswith('<0x') and token.endswith('>')

def simple_encode(token_str, tokenizer):
    return tokenizer.encode(token_str, add_special_tokens=False)

def special_encode(token_str, tokenizer):
    shift = len(tokenizer.encode('1', add_special_tokens=False))
    tokens = tokenizer.encode('1' + token_str, add_special_tokens=False)
    return tokens[shift:]

def get_mean_vec(token, tokenizer, embeddings, encode_func):
    tokens = encode_func(token, tokenizer)
    vector = embeddings[tokens].mean(axis=0)
    return vector

def convert_token_universal(token_str, tokenizer, vocab, tokenizer_prop):
    assert tokenizer.is_fast
    pre_tokenizer = tokenizer._tokenizer.pre_tokenizer
    if pre_tokenizer is not None:
        token_str = pre_tokenizer.pre_tokenize_str(token_str)
        token_str = ''.join([t[0] for t in token_str])

    if tokenizer_prop['space'] == '▁':
        token_str = token_str.replace(' ', '▁')

    if if_hex(token_str) and token_str in vocab:
        return tokenizer.convert_tokens_to_ids([token_str])

    return [t.id for t in tokenizer._tokenizer.model.tokenize(token_str)]

def convert_token_to_string_universal(token, tokenizer_dst, tokeniser_src_vocab, tokenizer_dst_properties):
    if if_hex(token):
        if token in tokeniser_src_vocab:
            return token
        
        token = chr(convert_ascii_hex(token))
        if token in tokeniser_src_vocab:
            return token
        
    token = [token]
    if tokenizer_dst_properties['force_leading_space']:
        token = [tokenizer_dst_properties['space']] + token

    text_token = tokenizer_dst.convert_tokens_to_string(token)
    if len(text_token) == 1 and ord(text_token) == 65533:
        return token[-1]
    
    return text_token

def get_tokenizer_properties(tokenizer):
    leading_space = False
    space = None
    char = '1'
    tokens = tokenizer(char, add_special_tokens=False)['input_ids']
    if len(tokens) > 1:
        space = tokenizer.convert_ids_to_tokens([tokens[0]])[0]
        leading_space = True
    else:
        token_str = tokenizer.convert_ids_to_tokens(tokens)[0]
        if len(token_str) != 1:
            space = token_str[0]
            leading_space = True

    space_token = tokenizer('1 ', add_special_tokens=False)['input_ids']
    if leading_space:
        assert len(space_token) == 3
        space_token = space_token[2]
        assert tokenizer.convert_ids_to_tokens([space_token])[0] == space
    else:
        assert len(space_token) == 2
        space_token = space_token[1]
        if space is None:
            space = tokenizer.convert_ids_to_tokens([space_token])[0]
        assert tokenizer.convert_ids_to_tokens([space_token])[0] == space

    return {'force_leading_space': leading_space, 'space': space}
'''
def reinit_embeddings_with_head_llama3_extended(model, tokenizer_src, tokenizer_dst, mode='random', lm_head_init='tie'):
    assert lm_head_init in ['tie', 'hm']
    assert mode in ['random', 'mean', 'occ']
    assert model.lm_head.bias is None

    #tokenizer_dst.add_special_tokens({'bos_token': tokenizer_src.added_tokens_decoder[128000], 'eos_token': tokenizer_src.added_tokens_decoder[128001], 'additional_special_tokens': list(tokenizer_src.added_tokens_decoder.values())})
    
    vocab_size = len(tokenizer_dst.get_vocab())
    vocab_old = tokenizer_src.get_vocab()
    #tokenizer_dst.vocab_size = vocab_size
    model.config.vocab_size = vocab_size

    embeddings_src = model.model.embed_tokens.weight.data.clone()
    lm_head_src = model.lm_head.weight.data.clone()
    
    model.model.embed_tokens = nn.Embedding(model.config.vocab_size, model.config.hidden_size, dtype=embeddings_src.dtype)
    model.lm_head = nn.Linear(model.config.hidden_size, model.config.vocab_size, bias=False, dtype=embeddings_src.dtype)

    if mode == 'random':
        model.model.embed_tokens.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        model.lm_head.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
    elif mode == 'mean':
        model.model.embed_tokens.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        model.lm_head.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        
        #spec_tokens = set(tokenizer_dst.special_tokens_map['additional_special_tokens']) if 'additional_special_tokens' in tokenizer_dst.special_tokens_map else set()
        #if 'eos_token' in tokenizer_dst.special_tokens_map:
        #    spec_tokens.add(tokenizer_dst.special_tokens_map['eos_token'])
        #if 'bos_token' in tokenizer_dst.special_tokens_map:
        #    spec_tokens.add(tokenizer_dst.special_tokens_map['bos_token'])
        #if 'unk_token' in tokenizer_dst.special_tokens_map:
        #    spec_tokens.add(tokenizer_dst.special_tokens_map['unk_token'])
        #if 'pad_token' in tokenizer_dst.special_tokens_map:
        #    spec_tokens.add(tokenizer_dst.special_tokens_map['pad_token'])
        
        zero_tokens_ids = set()
        for i in range(embeddings_src.shape[0]):
            if embeddings_src[i].norm() < 1e-12:
                print(i)
                zero_tokens_ids.add(i)

        print(len(zero_tokens_ids))
        with torch.no_grad():
            for i in tqdm(range(vocab_size)):
                #if i in zero_tokens_ids:
                #    model.model.embed_tokens.weight.data[i].copy_(embeddings_src[128009])
                #    model.lm_head.weight.data[i].copy_(lm_head_src[128009])
                #    continue
                token = tokenizer_dst._tokenizer.id_to_token(i)
                if token in vocab_old:
                    token_id = vocab_old[token]
                    if token_id in zero_tokens_ids:
                        print(i, token_id, '|'+token+'|')
                        model.model.embed_tokens.weight.data[i].copy_(embeddings_src[128009])
                        model.lm_head.weight.data[i].copy_(lm_head_src[128009])
                        continue
                    vec = embeddings_src[token_id]
                elif if_hex(token):
                    vec = get_hex_vec(token, tokenizer_src, embeddings_src)
                else:
                    token_string = tokenizer_dst.convert_tokens_to_string([token])
                    vec = get_mean_vec(token_string, tokenizer_src, embeddings_src, simple_encode) #vec = get_agg_vec(token, i, tokenizer_src, embeddings_src, mode)#get_mean_vec(token, tokenizer_src, embeddings_src)

                if vec.norm() < 1e-9:
                    print('skip ' + token)
                else:
                    model.model.embed_tokens.weight.data[i].copy_(vec)

                if lm_head_init == 'hm': # lm head mean
                    #vec = get_agg_vec(token, i, tokenizer_src, lm_head_src, mode)#get_mean_vec(token, tokenizer_src, lm_head_src)
                    if token in vocab_old:
                        vec = lm_head_src[token_id]
                    elif if_hex(token):
                        vec = get_hex_vec(token, tokenizer_src, lm_head_src)
                    else:
                        token_string = tokenizer_dst.convert_tokens_to_string([token])
                        vec = get_mean_vec(token_string, tokenizer_src, lm_head_src, simple_encode)
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

def get_hex_vec(token, tokenizer_src, embeddings):
    info = []
    info.append([token, tokenizer_src.vocab.get(token, -1)])
    if info[-1][1] != -1:
        return embeddings[info[-1][1]]
    token = chr(convert_ascii_hex(token))
    info.append([token, tokenizer_src.vocab.get(token, -1)])
    if info[-1][1] != -1:
        return embeddings[info[-1][1]]

    return get_mean_vec(token, tokenizer_src, embeddings, simple_encode)

def reinit_embeddings_with_head_llama3(model, tokenizer_src, tokenizer_dst, mode='random', lm_head_init='tie'):
    assert lm_head_init in ['tie', 'hm']
    assert mode in ['random', 'mean', 'occ']
    assert model.lm_head.bias is None

    tokenizer_dst.add_special_tokens({'bos_token': tokenizer_src.added_tokens_decoder[128000], 'eos_token': tokenizer_src.added_tokens_decoder[128001], 'additional_special_tokens': list(tokenizer_src.added_tokens_decoder.values())})
    
    vocab_size = len(tokenizer_dst.get_vocab())
    #tokenizer_dst.vocab_size = vocab_size
    model.config.vocab_size = vocab_size

    embeddings_src = model.model.embed_tokens.weight.data.clone()
    lm_head_src = model.lm_head.weight.data.clone()
    
    model.model.embed_tokens = nn.Embedding(model.config.vocab_size, model.config.hidden_size, dtype=embeddings_src.dtype)
    model.lm_head = nn.Linear(model.config.hidden_size, model.config.vocab_size, bias=False, dtype=embeddings_src.dtype)

    if mode == 'random':
        model.model.embed_tokens.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        model.lm_head.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
    elif mode == 'mean' or mode == 'occ':
        model.model.embed_tokens.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        model.lm_head.weight.data.normal_(mean=0.0, std=model.config.initializer_range)

        spec_tokens = set(tokenizer_dst.special_tokens_map['additional_special_tokens']) if 'additional_special_tokens' in tokenizer_dst.special_tokens_map else set()
        if 'eos_token' in tokenizer_dst.special_tokens_map:
            spec_tokens.add(tokenizer_dst.special_tokens_map['eos_token'])
        if 'bos_token' in tokenizer_dst.special_tokens_map:
            spec_tokens.add(tokenizer_dst.special_tokens_map['bos_token'])
        #if 'unk_token' in tokenizer_dst.special_tokens_map:
        #    spec_tokens.add(tokenizer_dst.special_tokens_map['unk_token'])
        if 'pad_token' in tokenizer_dst.special_tokens_map:
            spec_tokens.add(tokenizer_dst.special_tokens_map['pad_token'])

        with torch.no_grad():
            for i in tqdm(range(vocab_size)):
                token = tokenizer_dst._tokenizer.id_to_token(i)
                if token in spec_tokens:
                    vec = embeddings_src[tokenizer_src._tokenizer.token_to_id(token)]
                elif if_hex(token):
                    vec = get_hex_vec(token, tokenizer_src, embeddings_src)
                else:
                    token = token.replace('▁', ' ')
                    vec = get_mean_vec(token, tokenizer_src, embeddings_src, simple_encode)#get_agg_vec(token, i, tokenizer_src, embeddings_src, mode)#get_mean_vec(token, tokenizer_src, embeddings_src)

                model.model.embed_tokens.weight.data[i].copy_(vec)

                if lm_head_init == 'hm': # lm head mean
                    if token in spec_tokens:
                        vec = lm_head_src[tokenizer_src._tokenizer.token_to_id(token)]
                    elif if_hex(token):
                        vec = get_hex_vec(token, tokenizer_src, lm_head_src)
                    else:
                        vec = get_mean_vec(token, tokenizer_src, lm_head_src, simple_encode)#get_agg_vec(token, i, tokenizer_src, lm_head_src, mode)#get_mean_vec(token, tokenizer_src, lm_head_src)

                    model.lm_head.weight.data[i].copy_(vec)
                elif lm_head_init == 'tie':
                    model.lm_head.weight.data[i].copy_(vec)
                    
                #if lm_head_init == 'tie':
                #    model._tie_or_clone_weights(model.get_output_embeddings(), model.get_input_embeddings())
    else:
        raise Exception('NotImplemented')

    #print(model.model.embed_tokens.weight.data[0])
    #print(model.lm_head.weight.data[0])

def reinit_embeddings_with_head_mistral(model, tokenizer_src, tokenizer_dst, mode='random', lm_head_init='tie'):
    # Situation: 
    # dst tokenizer was trained via sentence piece and: 1) "space" =  '▁' 2) tokenizer forces space to first non-special token in sequence
    # src tokenizer shares these properties
    # mistral-7B-v0.1 as example

    assert lm_head_init in ['tie', 'hm']
    assert mode in ['random', 'mean']
    assert model.lm_head.bias is None

    # todo: check tokenizer properties

    tokenizer_src_prop = get_tokenizer_properties(tokenizer_src)
    tokenizer_dst_prop = get_tokenizer_properties(tokenizer_dst)


    vocab_size = len(tokenizer_dst.get_vocab())
    model.config.vocab_size = vocab_size

    embeddings_src = model.model.embed_tokens.weight.data.clone()
    lm_head_src = model.lm_head.weight.data.clone()
    
    model.model.embed_tokens = nn.Embedding(model.config.vocab_size, model.config.hidden_size, dtype=embeddings_src.dtype)
    model.lm_head = nn.Linear(model.config.hidden_size, model.config.vocab_size, bias=False, dtype=embeddings_src.dtype)

    model.model.embed_tokens.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
    model.lm_head.weight.data.normal_(mean=0.0, std=model.config.initializer_range)

    if mode == 'mean':
        spec_tokens = set(tokenizer_dst.special_tokens_map['additional_special_tokens']) if 'additional_special_tokens' in tokenizer_dst.special_tokens_map else set()
        if 'eos_token' in tokenizer_dst.special_tokens_map:
            spec_tokens.add(tokenizer_dst.special_tokens_map['eos_token'])
        if 'bos_token' in tokenizer_dst.special_tokens_map:
            spec_tokens.add(tokenizer_dst.special_tokens_map['bos_token'])
        if 'unk_token' in tokenizer_dst.special_tokens_map:
            spec_tokens.add(tokenizer_dst.special_tokens_map['unk_token'])
        if 'pad_token' in tokenizer_dst.special_tokens_map:
            spec_tokens.add(tokenizer_dst.special_tokens_map['pad_token'])

        with torch.no_grad():
            for i in tqdm(range(vocab_size)):
                token = tokenizer_dst._tokenizer.id_to_token(i)
                if token in spec_tokens:
                    vec = embeddings_src[tokenizer_src._tokenizer.token_to_id(token)]
                elif if_hex(token) and token in tokenizer_src.vocab:
                    idx = tokenizer_src.vocab[token]
                    vec = embeddings_src[idx]
                else:
                    vec = get_mean_vec(token, tokenizer_src, embeddings_src, special_encode)
                model.model.embed_tokens.weight.data[i].copy_(vec)

                if lm_head_init == 'hm': # lm head mean
                    if token in spec_tokens:
                        vec = lm_head_src[tokenizer_src._tokenizer.token_to_id(token)]
                    if if_hex(token) and token in tokenizer_src.vocab:
                        idx = tokenizer_src.vocab[token]
                        vec = lm_head_src[idx]
                    else:
                        vec = get_mean_vec(token, tokenizer_src, lm_head_src,special_encode)
                    model.lm_head.weight.data[i].copy_(vec)
                elif lm_head_init == 'tie':
                    model.lm_head.weight.data[i].copy_(vec)
    elif mode == 'random':
        pass
    else:
        raise Exception('NotImplemented')

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
        assert 'additional_special_tokens' not in tokenizer_dst.special_tokens_map or 'additional_special_tokens' not in tokenizer_src.special_tokens_map
        special_tokens_map = {key: val for key, val in tokenizer_src.special_tokens_map.items() if key != 'additional_special_tokens'}
        special_tokens_map['additional_special_tokens'] = list(tokenizer_src.added_tokens_decoder.values())
        tokenizer_dst.add_special_tokens(special_tokens_map)

    vocab_size = len(tokenizer_dst.get_vocab())
    print(f'New vocab size (maybe without new special tokens): {vocab_size}')
    model.config.vocab_size = vocab_size
    torch_dtype = model.model.embed_tokens.dtype
    print(f'dtype = {torch_dtype}')

    embeddings_src = model.model.embed_tokens.weight.data.clone().to(torch.float32)
    lm_head_src = model.lm_head.weight.data.clone().to(torch.float32)
    
    model.model.embed_tokens = nn.Embedding(model.config.vocab_size, model.config.hidden_size, dtype=torch_dtype)
    model.lm_head = nn.Linear(model.config.hidden_size, model.config.vocab_size, bias=False, dtype=torch_dtype)

    model.model.embed_tokens.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
    model.lm_head.weight.data.normal_(mean=0.0, std=model.config.initializer_range)

    if mode == 'mean':
        spec_tokens = set()
        for key in tokenizer_dst.special_tokens_map:
            if key == 'additional_special_tokens':
                spec_tokens.update(set(tokenizer_dst.special_tokens_map['additional_special_tokens']))
            else:
                spec_tokens.add(tokenizer_dst.special_tokens_map[key])

        with torch.no_grad():
            input_emb_mean = embeddings_src[:mean_cutoff].mean(axis=0).to(torch_dtype)
            output_emb_mean = lm_head_src[:mean_cutoff].mean(axis=0).to(torch_dtype)

            for i in tqdm(range(vocab_size)):
                token = tokenizer_dst._tokenizer.id_to_token(i)
                if token in spec_tokens:
                    embed_tokens_ids = [tokenizer_src._tokenizer.token_to_id(token)]
                else:
                    token_str = convert_token_to_string_universal(token, tokenizer_src, tokenizer_src_prop)
                    embed_tokens_ids = convert_token_universal(token_str, tokenizer_src)

                input_emb_vec = embeddings_src[embed_tokens_ids].mean(axis=0).to(torch_dtype)
                if input_emb_vec.norm() < 1e-12:
                    input_emb_vec = input_emb_mean.clone()
                model.model.embed_tokens.weight.data[i].copy_(input_emb_vec)

                if lm_head_init == 'hm':
                    output_emb_vec = lm_head_src[embed_tokens_ids].mean(axis=0).to(torch_dtype)
                elif lm_head_init == 'tie':
                    output_emb_vec = embeddings_src[embed_tokens_ids].mean(axis=0).to(torch_dtype)
                
                if output_emb_vec.norm() < 1e-12:
                    output_emb_vec = output_emb_mean.clone()
                model.lm_head.weight.data[i].copy_(output_emb_vec)

    elif mode == 'random':
        pass
    else:
        raise Exception('NotImplemented')
'''