import json
import torch
from transformers import AutoTokenizer
from .ushanka_proj_utils import PROJECTION_MODES


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


def has_alnum(text):
    return any(c.isalpha() for c in text)


def get_nonalnum_toks(tokenizer):
    return [tok for i in range(tokenizer.vocab_size) if not (has_alnum((tok := tokenizer.decode(i))))]


def init_new_tokenizer(model_tokenizer, donor_tokenizer, retain_nonalnum=False):
    new_tokenizer = AutoTokenizer.from_pretrained(donor_tokenizer.name_or_path)

    additional_vocab = list(
        model_tokenizer.get_added_vocab().keys()-donor_tokenizer.get_added_vocab().keys())

    non_alnum = []
    if retain_nonalnum:
        non_alnum = get_nonalnum_toks(model_tokenizer)

    if additional_vocab:
        new_tokenizer.add_tokens(additional_vocab)
        # print(additional_vocab)

    if non_alnum:
        new_tokenizer.add_tokens(non_alnum)

    new_tokenizer.add_special_tokens(model_tokenizer.special_tokens_map)
    new_tokenizer.chat_template = model_tokenizer.chat_template
    return new_tokenizer


def make_new_emb_matrix(module, new_size):
    tmp = module
    if "Embedd" in module.__class__.__name__:
        return tmp.__class__(new_size, tmp.weight.shape[-1], dtype=tmp.weight.dtype, device=tmp.weight.device)
    else:
        # LM-head
        return tmp.__class__(tmp.weight.shape[-1], new_size, dtype=tmp.weight.dtype, device=tmp.weight.device, bias=False)


def init_added_toks(embeddings_new, embeddings_old, tokenizer_new, tokenizer_old):
    for tok, tok_id in tokenizer_new.get_added_vocab().items():
        orig_id = tokenizer_old.encode(tok, add_special_tokens=False)
        print(tok, f"{orig_id}->{tok_id}")
        if len(orig_id) == 1:
            orig_id = orig_id[0]
            embeddings_new.weight[tok_id] = embeddings_old.weight[orig_id].detach()


def ushanka_embedding_projection(target_model, source_model, donor_model,
                                 model_tokenizer, donor_tokenizer, new_tokenizer,
                                 module_projection_modes=None, coocurrence_map_path=None,
                                 overlap_penalty=1.0):
    if module_projection_modes is None:
        module_projection_modes = {}
    # Fill missing entries with DEFAULT_PROJECTION_MODE
    for mod in EMB_MODULES:
        module_projection_modes[mod] = module_projection_modes.get(
            mod, DEFAULT_PROJECTION_MODE
        )

    base_vocab = donor_tokenizer.get_vocab()

    with torch.no_grad():
        for module in EMB_MODULES:
            print(module.upper())
            mod = target_model.get_submodule(module)
            src_mod = source_model.get_submodule(module)
            donor_mod = donor_model.get_submodule(module)

            new_mod = make_new_emb_matrix(mod, len(new_tokenizer.get_vocab()))
            print(new_mod.weight.shape, len(new_tokenizer.get_vocab()), new_mod.weight.device)
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
            
            print("Copying new embs...")
            new_mod.weight[:len(base_vocab)] = new_embs[:len(base_vocab)]

            # copy embs of service tokens
            print("Copying service tokens...")
            init_added_toks(new_mod, mod, new_tokenizer, model_tokenizer)
            mod.weight = new_mod.weight


def make_ushanka(target_model, source_model, donor_model,
                 model_tokenizer, donor_tokenizer, retain_nonalnum=False,
                 module_projection_modes=None, coocurrence_map_path=None,
                 overlap_penalty=1.0):
    new_tokenizer = init_new_tokenizer(
        model_tokenizer, donor_tokenizer, retain_nonalnum
    )
    ushanka_embedding_projection(
        target_model, source_model, donor_model,
        model_tokenizer, donor_tokenizer, new_tokenizer,
        module_projection_modes=module_projection_modes,
        coocurrence_map_path=coocurrence_map_path,
        overlap_penalty=overlap_penalty
    )
    return target_model, new_tokenizer
