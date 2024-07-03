from transformers import LlamaTokenizer, AutoTokenizer
import argparse
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path')
    parser.add_argument('--output_path')
    parser.add_argument('--version')
    args = parser.parse_args()

    tokenizer_class = LlamaTokenizer
    if args.version == 'v1':
        llama_tokenizer = tokenizer_class(args.model_path, sp_model_kwargs={'model_file': args.model_path}, legacy=True)
        llama_tokenizer.init_kwargs['sp_model_kwargs'] = {}
        llama_tokenizer.save_pretrained(args.output_path)
    elif args.version == 'v2':
        sp_model = spm.SentencePieceProcessor()
        sp_model.Load(args.model_path)

        spm = sp_pb2_model.ModelProto()
        spm.ParseFromString(sp_model.serialized_model_proto())

        output_sp_dir = args.output_path
        output_hf_dir =  args.output_path# the path to save Chinese-LLaMA tokenizer
        os.makedirs(output_sp_dir, exist_ok=True)
        with open(output_sp_dir+'/llama.model', 'wb') as f:
            f.write(spm.SerializeToString())

        tokenizer = tokenizer_class(vocab_file=output_sp_dir+'/llama.model')
        tokenizer.save_pretrained(output_hf_dir)

