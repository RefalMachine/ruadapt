import json

import torch

from .projection import PROJECTION_MODES


def load_occs(path):
    with open(path, encoding='utf-8') as f:
        return {int(k): {int(kk): int(vv) for kk, vv in v.items()}for k, v in json.load(f).items()}


def convert_tok_map_to_probs(tok_map):
    for k, v in tok_map.items():
        norm = sum(v.values())
        for vv in v.keys():
            tok_map[k][vv] /= norm


DEFAULT_PROJECTION_MODE = "conversion"  # "occ_conversion"
EMB_MODULES = ["lm_head", "model.embed_tokens"]
INPUT_EMBED_MODULE = "model.embed_tokens"
OUTPUT_EMBED_MODULE = "lm_head"


def has_alnum(text):
    return any(c.isalpha() for c in text)


def get_nonalnum_toks(tokenizer):
    return [tok for i in range(tokenizer.vocab_size) if not (has_alnum((tok := tokenizer.decode(i))))]


def _token_to_string(token):
    return str(getattr(token, "content", token))


def _special_token_strings(tokenizer):
    tokens = []
    for token in getattr(tokenizer, "all_special_tokens", []) or []:
        token = _token_to_string(token)
        if token not in tokens:
            tokens.append(token)
    return tokens


def _special_tokens_map_as_strings(tokenizer):
    result = {}
    for key, value in tokenizer.special_tokens_map.items():
        if isinstance(value, (list, tuple)):
            result[key] = [_token_to_string(token) for token in value]
        else:
            result[key] = _token_to_string(value)
    return result


def apply_target_special_tokens(new_tokenizer, target_tokenizer):
    special_tokens_map = _special_tokens_map_as_strings(target_tokenizer)
    additional = special_tokens_map.pop("additional_special_tokens", [])
    if special_tokens_map:
        new_tokenizer.add_special_tokens(special_tokens_map)
    if additional:
        try:
            new_tokenizer.add_special_tokens(
                {"additional_special_tokens": additional}, replace_additional_special_tokens=True
            )
        except TypeError:
            new_tokenizer.add_special_tokens({"additional_special_tokens": additional})
        new_tokenizer.additional_special_tokens = additional

    for attr_name in ("bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token"):
        token = getattr(target_tokenizer, attr_name, None)
        if token is not None:
            setattr(new_tokenizer, attr_name, _token_to_string(token))


def init_new_tokenizer(model_tokenizer, donor_tokenizer, retain_nonalnum=False):
    from transformers import AutoTokenizer

    new_tokenizer = AutoTokenizer.from_pretrained(donor_tokenizer.name_or_path)

    model_special_tokens = set(_special_token_strings(model_tokenizer))
    additional_vocab = list(
        model_tokenizer.get_added_vocab().keys()-donor_tokenizer.get_added_vocab().keys())
    additional_vocab = [tok for tok in additional_vocab if tok not in model_special_tokens]

    non_alnum = []
    if retain_nonalnum:
        non_alnum = get_nonalnum_toks(model_tokenizer)

    if additional_vocab:
        new_tokenizer.add_tokens(additional_vocab)
        # print(additional_vocab)

    if non_alnum:
        new_tokenizer.add_tokens(non_alnum)

    apply_target_special_tokens(new_tokenizer, model_tokenizer)
    new_tokenizer.chat_template = model_tokenizer.chat_template
    return new_tokenizer


def make_new_emb_matrix(module, new_size):
    tmp = module
    if "Embedd" in module.__class__.__name__:
        return tmp.__class__(new_size, tmp.weight.shape[-1], dtype=tmp.weight.dtype, device=tmp.weight.device)
    else:
        # LM-head
        return tmp.__class__(tmp.weight.shape[-1], new_size, dtype=tmp.weight.dtype, device=tmp.weight.device, bias=False)


def get_tokenizer_matrix_size(tokenizer):
    vocab = tokenizer.get_vocab()
    return max(vocab.values()) + 1 if vocab else 0


def get_emb_module(model, module_name):
    if module_name == INPUT_EMBED_MODULE:
        module = model.get_input_embeddings()
    elif module_name == OUTPUT_EMBED_MODULE:
        module = model.get_output_embeddings()
    else:
        module = model.get_submodule(module_name)
    if module is None:
        raise AttributeError(f"Cannot find {module_name} on {type(model).__name__}")
    return module


def retie_embeddings(model):
    input_embeddings = get_emb_module(model, INPUT_EMBED_MODULE)
    output_embeddings = get_emb_module(model, OUTPUT_EMBED_MODULE)
    output_embeddings.weight = input_embeddings.weight
    if hasattr(model, "tie_weights"):
        model.tie_weights()


def init_added_toks(embeddings_new, embeddings_old, tokenizer_new, tokenizer_old):
    for tok, tok_id in tokenizer_new.get_added_vocab().items():
        orig_id = tokenizer_old.encode(tok, add_special_tokens=False)
        print(tok, f"{orig_id}->{tok_id}")
        if len(orig_id) == 1:
            orig_id = orig_id[0]
            embeddings_new.weight[tok_id] = embeddings_old.weight[orig_id].detach()


def get_tokens_transfer_plan(new_tokenizer, target_tokenizer, donor_tokenizer):
    final_vocab = new_tokenizer.get_vocab()
    target_vocab = target_tokenizer.get_vocab()
    donor_vocab = donor_tokenizer.get_vocab()
    preserved = {}
    projected = {}
    missing_in_donor = []
    for token, final_id in final_vocab.items():
        if token in target_vocab:
            preserved[token] = (final_id, target_vocab[token])
        elif token in donor_vocab:
            projected[token] = (final_id, donor_vocab[token])
        else:
            missing_in_donor.append(token)
    if missing_in_donor:
        raise ValueError(f"Tokens absent from donor tokenizer: {missing_in_donor[:20]}")
    return preserved, projected


def validate_special_tokens_present(new_tokenizer, target_tokenizer):
    final_vocab = new_tokenizer.get_vocab()
    target_vocab = target_tokenizer.get_vocab()
    missing_in_final = [token for token in _special_token_strings(target_tokenizer) if token not in final_vocab]
    missing_in_target = [token for token in _special_token_strings(target_tokenizer) if token not in target_vocab]
    if missing_in_final:
        raise ValueError(f"Special tokens absent from final tokenizer: {missing_in_final}")
    if missing_in_target:
        raise ValueError(f"Special tokens absent from target tokenizer vocab: {missing_in_target}")


def copy_token_embeddings_by_plan(new_mod, target_mod, projected_donor_embs, preserved, projected):
    for token, (final_id, target_id) in preserved.items():
        if final_id >= new_mod.weight.shape[0] or target_id >= target_mod.weight.shape[0]:
            raise IndexError(f"Preserved token id out of range for {token}: {target_id}->{final_id}")
        new_mod.weight[final_id] = target_mod.weight[target_id].detach().to(
            device=new_mod.weight.device, dtype=new_mod.weight.dtype
        )
    for token, (final_id, donor_id) in projected.items():
        if final_id >= new_mod.weight.shape[0] or donor_id >= projected_donor_embs.shape[0]:
            raise IndexError(f"Projected token id out of range for {token}: {donor_id}->{final_id}")
        new_mod.weight[final_id] = projected_donor_embs[donor_id].detach().to(
            device=new_mod.weight.device, dtype=new_mod.weight.dtype
        )


def ushanka_embedding_projection(target_model, source_model, donor_model,
                                 model_tokenizer, donor_tokenizer, new_tokenizer,
                                 module_projection_modes=None, coocurrence_map_path=None,
                                 overlap_penalty=1.0, project_added_tokens_only=False):
    if module_projection_modes is None:
        module_projection_modes = {}
    # Fill missing entries with DEFAULT_PROJECTION_MODE
    
    assert target_model.config.tie_word_embeddings == source_model.config.tie_word_embeddings == donor_model.config.tie_word_embeddings
    emb_modules = EMB_MODULES
    if target_model.config.tie_word_embeddings:
        emb_modules = [INPUT_EMBED_MODULE]
        print('Models have tie embeddings. LEP only input embeddings')
        
    for mod in emb_modules:
        module_projection_modes[mod] = module_projection_modes.get(
            mod, DEFAULT_PROJECTION_MODE
        )

    base_vocab = donor_tokenizer.get_vocab()
    if project_added_tokens_only:
        validate_special_tokens_present(new_tokenizer, model_tokenizer)
        preserved, projected = get_tokens_transfer_plan(new_tokenizer, model_tokenizer, donor_tokenizer)
        print(f"Token transfer plan: {len(preserved)} preserved target tokens, {len(projected)} projected donor tokens")

    with torch.no_grad():
        for module in emb_modules:
            print(module.upper())
            mod = get_emb_module(target_model, module)
            src_mod = get_emb_module(source_model, module)
            donor_mod = get_emb_module(donor_model, module)

            new_vocab_size = get_tokenizer_matrix_size(new_tokenizer)
            new_mod = make_new_emb_matrix(mod, new_vocab_size)
            print(new_mod.weight.shape, new_vocab_size, new_mod.weight.device)
            print(donor_mod.weight.shape)

            # all projs happen here
            if module_projection_modes[module] in PROJECTION_MODES:
                print("Making projection...")
                proj = PROJECTION_MODES[module_projection_modes[module]](
                    src_embs=src_mod.weight, tgt_embs=mod.weight,
                    old_tokenizer=model_tokenizer, new_tokenizer=donor_tokenizer,
                    coocurrence_map_path=coocurrence_map_path,
                    overlap_penalty=overlap_penalty
                )
                print("Applying projection...")
                new_embs = donor_mod.weight.detach() @ proj.to(donor_mod.weight.device)
            else:
                raise ValueError

            if project_added_tokens_only:
                print("Copying preserved target tokens and projected donor-only tokens...")
                copy_token_embeddings_by_plan(new_mod, mod, new_embs, preserved, projected)
            else:
                print("Copying new embs...")
                new_mod.weight[:len(base_vocab)] = new_embs[:len(base_vocab)]
                print("Copying service tokens...")
                init_added_toks(new_mod, mod, new_tokenizer, model_tokenizer)
            mod.weight = new_mod.weight

        if target_model.config.tie_word_embeddings:
            retie_embeddings(target_model)



def make_ushanka(target_model, source_model, donor_model,
                 model_tokenizer, donor_tokenizer, retain_nonalnum=False,
                 module_projection_modes=None, coocurrence_map_path=None,
                 overlap_penalty=1.0, project_added_tokens_only=False):
    new_tokenizer = init_new_tokenizer(
        model_tokenizer, donor_tokenizer, retain_nonalnum
    )
    ushanka_embedding_projection(
        target_model, source_model, donor_model,
        model_tokenizer, donor_tokenizer, new_tokenizer,
        module_projection_modes=module_projection_modes,
        coocurrence_map_path=coocurrence_map_path,
        overlap_penalty=overlap_penalty,
        project_added_tokens_only=project_added_tokens_only,
    )
    return target_model, new_tokenizer
