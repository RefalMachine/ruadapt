from torch import nn
from tqdm import tqdm
import torch

def special_encode(token_str, tokenizer):
    shift = len(tokenizer.encode('家', add_special_tokens=False))
    tokens = tokenizer.encode('家' + token_str, add_special_tokens=False)
    return tokens[shift:]

def get_mean_vec(token, tokenizer, embeddings):
    tokens = special_encode(token, tokenizer)
    vector = embeddings[tokens].mean(axis=0)
    return vector

def reinit_embeddings_with_head_llama3(model, tokenizer_old, tokenizer_new, mode='random', lm_head_init='tie'):
    assert lm_head_init in ['tie', 'hm']
    assert mode in ['random', 'mean']
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
    elif mode == 'mean':
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
                else:
                    token = token.replace('▁', ' ')
                    vec = get_mean_vec(token, tokenizer_old, embeddings_old)
                model.model.embed_tokens.weight.data[i].copy_(vec)

                if lm_head_init == 'hm': # lm head mean
                    if token in spec_tokens:
                        vec = lm_head_old[tokenizer_old._tokenizer.token_to_id(token)]
                    else:
                        vec = get_mean_vec(token, tokenizer_old, lm_head_old)
                    model.lm_head.weight.data[i].copy_(vec)
                elif lm_head_init == 'tie':
                    model.lm_head.weight.data[i].copy_(vec)
                    
                #if lm_head_init == 'tie':
                #    model._tie_or_clone_weights(model.get_output_embeddings(), model.get_input_embeddings())
    else:
        raise Exception('NotImplemented')

    print(model.model.embed_tokens.weight.data[0])
    print(model.lm_head.weight.data[0])

def reinit_embeddings_with_head_llama(model, tokenizer_old, tokenizer_new, mode='random', lm_head_init='tie', leading_space_llama3=False):
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
                if leading_space_llama3:
                    token = token.replace('▁', ' ')
                vec = get_mean_vec(token, tokenizer_old, embeddings_old)
                model.model.embed_tokens.weight.data[i].copy_(vec)

                if lm_head_init == 'hm': # lm head mean
                    vec = get_mean_vec(token, tokenizer_old, lm_head_old)
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