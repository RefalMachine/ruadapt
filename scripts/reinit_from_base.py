import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftConfig, PeftModel
import tiktoken
import tiktoken_ext.openai_public

def reinit_from_base(dst_model_path, src_model_path, out_model_path, tiktoken_extended_path):
    mergeable_ranks_extend = tiktoken_ext.openai_public.load_tiktoken_bpe(tiktoken_extended_path)
    new_idx = list(mergeable_ranks_extend.values())
    old_idx = list(range(0, min(new_idx)))
    print(len(old_idx))

    tokenizer = AutoTokenizer.from_pretrained(dst_model_path)
    dst_model = AutoModelForCausalLM.from_pretrained(
        dst_model_path,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        device_map='cpu',
    )

    src_model = AutoModelForCausalLM.from_pretrained(
        src_model_path,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        device_map='cpu',
    )

    for idx in [0, 10000, 50000, 100000, 120000]:
        print(idx)
        print(dst_model.model.embed_tokens.weight[idx])
        print(src_model.model.embed_tokens.weight[idx])
        print()

    #print(dst_model.lm_head.weight[50000])
    #print(dst_model.model.embed_tokens.weight[min(new_idx)])
    with torch.no_grad():
        dst_model.model.embed_tokens.weight.data[old_idx] = src_model.model.embed_tokens.weight.data[old_idx]
        if not dst_model.config.tie_word_embeddings:
            dst_model.lm_head.weight.data[old_idx] = src_model.model.lm_head.weight.data[old_idx]

    for idx in [0, 10000, 50000, 100000, 120000]:
        print(idx)
        print(dst_model.model.embed_tokens.weight[idx])
        print(dst_model.lm_head.weight[idx])
        print(src_model.model.embed_tokens.weight[idx])
        
        print()
    #print(dst_model.model.embed_tokens.weight[50000])
    #print(dst_model.lm_head.weight[50000])
    #print(dst_model.model.embed_tokens.weight[min(new_idx)])

    tokenizer.save_pretrained(out_model_path)
    dst_model.save_pretrained(out_model_path)
    
if __name__ == "__main__":
    fire.Fire(reinit_from_base)
