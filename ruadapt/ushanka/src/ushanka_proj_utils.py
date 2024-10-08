import json
#from tqdm.auto import tqdm
from tqdm import tqdm
import random
import torch


def special_encode(token_str, tokenizer):
    if "llama" in tokenizer.name_or_path.lower():
        token_str = token_str.replace('▁', ' ')
    shift = len(tokenizer.encode('家', add_special_tokens=False))
    tokens = tokenizer.encode('家' + token_str, add_special_tokens=False)
    return tokens[shift:]


n_gram_model = 5

#was gelsy but must be set to gels for cuda
def get_proj_matrix(A, B, driver='gelsy'):
    print(f"Shape of A projection {A.shape}")
    with torch.no_grad():
        proj = torch.linalg.lstsq(A, B, driver=driver)
        print(proj.residuals)
        return proj.solution


def load_occs(path):
    with open(path, encoding='utf-8') as f:
        return {int(k): {int(kk): int(vv) for kk, vv in v.items()}for k, v in json.load(f).items()}


def convert_tok_map_to_probs(tok_map):
    for k, v in tok_map.items():
        norm = sum(v.values())
        for vv in v.keys():
            tok_map[k][vv] /= norm


def get_agg_vec(token, token_id, tokenizer, embeddings, agg_mode='mean',
                occ_tok_map=None, overlap_penalty=1.0, **kwargs):
    if agg_mode == 'occ' and token_id in occ_tok_map:
        subtokens, probs = list(zip(*sorted(occ_tok_map[token_id].items())))
        subtokens = list(subtokens)
        # print(subtokens)
    else:
        subtokens = special_encode(token, tokenizer)
    vector = embeddings[subtokens]
    if len(subtokens) > 1:
        if agg_mode == 'mean':
            vector = vector.mean(axis=0)
        elif agg_mode == 'sum':
            vector = vector.sum(axis=0)
        elif agg_mode == 'occ':
            if token_id in occ_tok_map:
                probs = torch.tensor(probs, dtype=vector.dtype).reshape(-1, 1)
                vector = (probs * vector).sum(axis=0)
            else:
                print(f"Token id {token_id} not found in occurance map")
                vector = vector.mean(axis=0)
        else:
            raise ValueError
    else:
        # print(overlap_penalty)
        vector = vector.reshape(-1) / overlap_penalty
    return vector


def convert_embs_to_new_tok(old_embs, old_tok, new_tok, agg_mode='mean', **kwargs):
    with torch.no_grad():
        res = []
        id_list = range(len(new_tok.vocab))
        for i in tqdm(id_list):
            token = new_tok._tokenizer.id_to_token(i)
            if i in new_tok.all_special_ids:
                old_id = old_tok._tokenizer.token_to_id(token)
                print(token, i, old_id)
                if old_id is not None:
                    vec = old_embs[old_id]
                else:
                    print(f"Skipped token {token}")
                    continue
            else:
                vec = get_agg_vec(token, i, old_tok, old_embs, agg_mode=agg_mode, **kwargs)
            res.append(vec)
        return torch.stack(res)


def make_conversion_proj(src_embs, tgt_embs, old_tokenizer, new_tokenizer, agg_mode='mean', **kwargs):
    print(agg_mode)
    E_src = convert_embs_to_new_tok(src_embs, old_tokenizer, new_tokenizer, agg_mode=agg_mode, **kwargs)
    E_tgt = convert_embs_to_new_tok(tgt_embs, old_tokenizer, new_tokenizer, agg_mode=agg_mode, **kwargs)

    proj = get_proj_matrix(
        E_src.cpu().to(torch.float64),
        E_tgt.cpu().to(torch.float64)
    ).to(src_embs.dtype)
    return proj


def mean_conv(src_embs, tgt_embs, old_tokenizer, new_tokenizer, **kwargs):
    return make_conversion_proj(src_embs, tgt_embs, old_tokenizer,
                                new_tokenizer, agg_mode='mean', **kwargs)


def instant_conv(src_embs, tgt_embs, old_tokenizer, new_tokenizer, **kwargs):
    return make_conversion_proj(src_embs, tgt_embs, old_tokenizer,
                                old_tokenizer, agg_mode='mean', **kwargs)

def sum_conv(src_embs, tgt_embs, old_tokenizer, new_tokenizer, **kwargs):
    return make_conversion_proj(src_embs, tgt_embs, old_tokenizer,
                                new_tokenizer, agg_mode='sum', **kwargs)


def occ_conv(src_embs, tgt_embs, old_tokenizer, new_tokenizer,
             coocurrence_map_path, **kwargs):
    occ_tok_map = load_occs(coocurrence_map_path)
    convert_tok_map_to_probs(occ_tok_map)
    return make_conversion_proj(
        src_embs, tgt_embs, old_tokenizer,
        new_tokenizer, agg_mode='occ', occ_tok_map=occ_tok_map, **kwargs
    )


def make_straight_sub_proj(src_embs, **kwargs):
    return torch.eye(src_embs.shape[-1]).to(dtype=src_embs.dtype)


# def make_full_proj(src_embs, tgt_embs, **kwargs):
#     # assume that all tokens of foundation source model
#     # are retained in fine-tune tgt model on the same positions
#     shared_keys_border = min(src_embs.shape[0], tgt_embs.shape[0])
#     E_src = src_embs[:shared_keys_border]
#     E_tgt = tgt_embs[:shared_keys_border]

#     proj = get_proj_matrix(
#         E_src.to(torch.float64),
#         E_tgt.to(torch.float64)
#     ).to(torch.bfloat16)
#     return proj


def get_union_tokens(old_tokenizer, new_tokenizer):
    old_readable_keys = {old_tokenizer.convert_tokens_to_string([k]): v for k,v in old_tokenizer.vocab.items()}
    new_readable_keys = {new_tokenizer.convert_tokens_to_string([k]): k for k in new_tokenizer.vocab.keys()}

    shared_keys = list(old_readable_keys.keys() & new_readable_keys.keys())
    shared_keys_ids = sorted([old_readable_keys[k] for k in shared_keys])
    return shared_keys_ids


def make_union_proj(src_embs, tgt_embs, old_tokenizer, new_tokenizer, **kwargs):
    shared_keys_ids = get_union_tokens(old_tokenizer, new_tokenizer)

    E_src = src_embs[shared_keys_ids]
    E_tgt = tgt_embs[shared_keys_ids]
    
    print(E_src.device)
    
    proj = get_proj_matrix(
        E_src.cpu().to(torch.float64),
        E_tgt.cpu().to(torch.float64)
    ).to(src_embs.dtype)
    return proj


# def make_update_projection_embs(src_embs, tgt_embs, donor_embs, old_tokenizer, new_tokenizer):
#     E_src = convert_embs_to_new_tok(src_embs, old_tokenizer, new_tokenizer)
#     E_tgt = convert_embs_to_new_tok(tgt_embs, old_tokenizer, new_tokenizer)

#     adaptation_proj = make_full_proj(E_src, donor_embs)

#     # shared_keys_border = min(src_embs.shape[0], tgt_embs.shape[0])
#     # update_delta=tgt_embs[:shared_keys_border]-src_embs[:shared_keys_border]
#     # update_delta=convert_embs_to_new_tok(update_delta, old_tokenizer, new_tokenizer)
#     update_delta = E_tgt-E_src

#     projected_delta = update_delta @ adaptation_proj

#     return donor_embs.detach() + projected_delta


PROJECTION_MODES = {
    "straight": make_straight_sub_proj,
    "union": make_union_proj,
    "conversion": mean_conv,
    "bypass":instant_conv,
    "sum_conversion": sum_conv,
    "occ_conversion": occ_conv,
}


def list_projection_modes():
    return list(PROJECTION_MODES.keys())
