from torch import nn
from tqdm import tqdm
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, AutoConfig
import torch
#from src.tokenization.utils import reinit_embeddings_with_head_llama3, special_encode, get_mean_vec

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_model_name_or_path')
    parser.add_argument('--donor_model_name_or_path')
    parser.add_argument('--output_path')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.source_model_name_or_path)
    print(tokenizer.bos_token)
    print(tokenizer.bos_token_id)
    config = AutoConfig.from_pretrained(args.source_model_name_or_path)
    src_model = AutoModelForCausalLM.from_pretrained(args.source_model_name_or_path, device_map='auto', torch_dtype=config.torch_dtype)
    donor_model = AutoModelForCausalLM.from_pretrained(args.donor_model_name_or_path, device_map='auto', torch_dtype=config.torch_dtype)

    print(src_model.lm_head.weight[13])
    print(donor_model.lm_head.weight[13])
    src_model.lm_head.weight.data.copy_(donor_model.lm_head.weight.detach().data)
    print(src_model.lm_head.weight[13])

    src_model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
