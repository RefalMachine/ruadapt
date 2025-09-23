import argparse
import os
import subprocess
import shutil

def file_exists_nonempty(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0

def dir_has_files(path: str) -> bool:
    return os.path.isdir(path) and any(os.scandir(path))

def touch(path: str):
    with open(path, 'a'):
        os.utime(path, None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--op')
    parser.add_argument('--src_model_path')
    parser.add_argument('--output_path')
    parser.add_argument('--replace_tokenizer_path', default='')
    parser.add_argument('--extend_tiktoken_tokenizer_path', default='')
    parser.add_argument('--extend_hf_tokenizer_path', default='')
    parser.add_argument('--extend_hf_tokenizer_type', default='')
    parser.add_argument('--only_ru', action='store_true')
    parser.add_argument('--filter_numbers', action='store_true')
    parser.add_argument('--custom_tokens_path', default=None)
    parser.add_argument('--init_mode', default='mean')
    parser.add_argument('--mult', default=1.0, type=float)
    args = parser.parse_args()

    assert args.op in ['replace', 'extend']

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if args.op == 'replace':
        replace_marker = os.path.join(args.output_path, '.replace_done')
        if os.path.exists(replace_marker):
            print('[SKIP] replace: completed step found')
        else:
            call_res = subprocess.call(['python', '-m', 'ruadapt.tokenization.run_replace_tokenizer', 
                            '--model_name_or_path', args.src_model_path,
                            '--new_tokenizer_path', args.replace_tokenizer_path,
                            '--output_path', args.output_path])
            if call_res == 0:
                touch(replace_marker)
            else:
                print(f"[ERROR] Stopping pipeline at replace step. Error code: {call_res}")
                exit(1)

    elif args.op == 'extend':
        sp_tok_vocab_freq_path = os.path.join(args.output_path, 'sp_tok_vocab_freq.txt')
        
        if file_exists_nonempty(sp_tok_vocab_freq_path):
            print(f"[SKIP] Skipping convert_hf_tokenizer_vocab_to_freq_list, results already exists")
        else:
            call_params = ['python', '-m', 'ruadapt.tokenization.convert_hf_tokenizer_vocab_to_freq_list', 
                '--tokenizer_path', args.extend_hf_tokenizer_path,
                '--output_path', sp_tok_vocab_freq_path,
                '--type', args.extend_hf_tokenizer_type]
        
            if args.custom_tokens_path is not None:
                call_params += ['--custom_tokens_path', args.custom_tokens_path]
            
            if args.only_ru:
                call_params.append('--only_ru')

            
            call_res = subprocess.call(call_params)
            if call_res != 0 or not file_exists_nonempty(sp_tok_vocab_freq_path):
                print(call_res)
                print('ERROR. Stoping pipeline at convert_hf_tokenizer_vocab_to_freq_list')
                exit(1)
        
        tokenizer_extended_part_path = os.path.join(args.output_path, 'tokenizer_extended_part.tiktoken')
        tiktoken_base_path = os.path.join(args.output_path, 'tokenizer_base.tiktoken')
        
        if file_exists_nonempty(tiktoken_base_path):
            print(f"[SKIP] Skipping, base tiktoken already exists")
        else:
            shutil.copyfile(args.extend_tiktoken_tokenizer_path, tiktoken_base_path)
            if not file_exists_nonempty(tiktoken_base_path):
                print('[ERROR] Base tiktoken copy failed')
                exit(1)

        if file_exists_nonempty(tokenizer_extended_part_path):
            print("[SKIP] Skipping add_merges call, extended tiktoken already exists")
        else:
            call_res = subprocess.call(
                ['python', '-m', 'ruadapt.tokenization.add_merges', 
                '--input_path', tiktoken_base_path,
                '--output_path', tokenizer_extended_part_path,
                '--vocab_path', sp_tok_vocab_freq_path,
                '--start_id', str(-1)])
        
            if call_res != 0 or not file_exists_nonempty(tokenizer_extended_part_path):
                print(call_res)
                print('ERROR. Stoping pipeline')
                exit(1)
        
        hf_tokenizer_dir = os.path.join(args.output_path, 'hf_tokenizer')
        if dir_has_files(hf_tokenizer_dir):
            print('[SKIP] Skipping expand_tiktoken_save_hf, dir already non empty')
        else:

            call_params = ['python', '-m', 'ruadapt.tokenization.expand_tiktoken_save_hf', 
                '--tiktoken_base_path', tiktoken_base_path,
                '--tiktoken_new_path', tokenizer_extended_part_path,
                '--output_dir', hf_tokenizer_dir,
                '--init_output_from', args.src_model_path]  
            if args.filter_numbers:
                call_params.append('--filter_numbers')

            call_res = subprocess.call(call_params)
                
            if call_res != 0 or not dir_has_files(hf_tokenizer_dir):
                print(call_res)
                print('ERROR. Stoping pipeline')
                exit(1)
            
        final_replace_marker = os.path.join(args.output_path, '.extend_replace_done')
        if os.path.exists(final_replace_marker):
            print('[SKIP] final replace already done')
        else:
            call_res = subprocess.call(
                ['python', '-m', 'ruadapt.tokenization.run_replace_tokenizer', 
                '--model_name_or_path', args.src_model_path,
                '--new_tokenizer_path', os.path.join(args.output_path, 'hf_tokenizer'),
                '--output_path', args.output_path,
                '--mode', args.init_mode,
                '--mult', str(args.mult)])

            if call_res != 0:
                print(call_res)
                print('ERROR. Stopping pipeline at final replace step')
                exit(1)
            touch(final_replace_marker)

        
        

