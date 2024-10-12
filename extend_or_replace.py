import argparse
import os
import subprocess
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--op')
    parser.add_argument('--src_model_path')
    parser.add_argument('--output_path')
    parser.add_argument('--replace_tokenizer_path', default='')
    parser.add_argument('--extend_tiktoken_tokenizer_path', default='')
    parser.add_argument('--extend_hf_tokenizer_path', default='')
    parser.add_argument('--extend_hf_tokenizer_type', default='')
    args = parser.parse_args()

    assert args.op in ['replace', 'extend']

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if args.op == 'replace':
        subprocess.call(['python', '-m', 'ruadapt.tokenization.run_replace_tokenizer', 
                         '--model_name_or_path', args.src_model_path,
                         '--new_tokenizer_path', args.replace_tokenizer_path,
                         '--output_path', args.output_path])
    elif args.op == 'extend':
        sp_tok_vocab_freq_path = os.path.join(args.output_path, 'sp_tok_vocab_freq.txt')
        call_res = subprocess.call(
            ['python', '-m', 'ruadapt.tokenization.convert_hf_tokenizer_vocab_to_freq_list', 
             '--tokenizer_path', args.extend_hf_tokenizer_path,
             '--output_path', sp_tok_vocab_freq_path,
             '--type', args.extend_hf_tokenizer_type])
        
        if call_res != 0:
            print(call_res)
            print('ERROR. Stoping pipeline')
            exit(1)
        
        tokenizer_extended_part_path = os.path.join(args.output_path, 'tokenizer_extended_part.tiktoken')
        tiktoken_base_path = os.path.join(args.output_path, 'tokenizer_base.tiktoken')
        
        shutil.copyfile(args.extend_tiktoken_tokenizer_path, tiktoken_base_path)
        call_res = subprocess.call(
            ['python', '-m', 'ruadapt.tokenization.add_merges', 
             '--input_path', tiktoken_base_path,
             '--output_path', tokenizer_extended_part_path,
             '--vocab_path', sp_tok_vocab_freq_path,
             '--start_id', str(-1)])
        
        if call_res != 0:
            print(call_res)
            print('ERROR. Stoping pipeline')
            exit(1)
        
        call_res = subprocess.call(
            ['python', '-m', 'ruadapt.tokenization.expand_tiktoken_save_hf', 
             '--tiktoken_base_path', tiktoken_base_path,
             '--tiktoken_new_path', tokenizer_extended_part_path,
             '--output_dir', os.path.join(args.output_path, 'hf_tokenizer'),
             '--init_output_from', args.src_model_path])
        
        if call_res != 0:
            print(call_res)
            print('ERROR. Stoping pipeline')
            exit(1)
            
        subprocess.call(
            ['python', '-m', 'ruadapt.tokenization.run_replace_tokenizer', 
             '--model_name_or_path', args.src_model_path,
             '--new_tokenizer_path', os.path.join(args.output_path, 'hf_tokenizer'),
             '--output_path', args.output_path])

        
        

