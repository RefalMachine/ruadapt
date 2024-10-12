import argparse
import os
import subprocess
import shutil
import codecs
import json
from transformers import AutoTokenizer

def resolve_special_tokens(model_path, bos_token, eos_token, pad_token):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if bos_token is None:
        #bos_token = tokenizer.config.bos_token
        #if bos_token is None:
        bos_token = tokenizer.bos_token
        if bos_token is None:
            bos_token = tokenizer.special_tokens_map.get('bos_token', None)

    if eos_token is None:
        #eos_token = tokenizer.config.eos_token
        #if eos_token is None:
        eos_token = tokenizer.eos_token
        if eos_token is None:
            eos_token = tokenizer.special_tokens_map.get('eos_token', None)

    assert bos_token is not None or eos_token is not None
    if bos_token is None:
        bos_token = eos_token
    elif eos_token is None:
        eos_token = bos_token

    if pad_token is None:
        pad_token = bos_token

    print(f'ATTENTION:\n\tbos_token={bos_token}\n\teos_token={eos_token}\n\tpad_token={pad_token}\nIf it is incorrect, then use custom_tokens parameters')
    return {'bos_token': bos_token, 'eos_token': eos_token, 'pad_token': pad_token}

def create_lep_config(target_model_path, source_model_path, donor_model_path, output_dir):
    lep_model_path = os.path.join(output_dir, 'lep')
    if not os.path.exists(lep_model_path):
        os.mkdir(lep_model_path)

    lep_config = {
        "target_model_path" : target_model_path,
        "source_model_path" : source_model_path,
        "donor_model_path" : donor_model_path
    }

    lep_config_path = os.path.join(lep_model_path, 'lep_config.json')
    with codecs.open(lep_config_path, 'w', 'utf-8') as file:
        json.dump(lep_config, file, ensure_ascii=False, indent=4)

    return lep_model_path, lep_config_path

def create_step_config(step_idx, step, model_name_or_path, output_dir, special_tokens):
    step_model_path = os.path.join(output_dir, step['type'] + str(step_idx) + '_lora')
    if not os.path.exists(step_model_path):
        os.mkdir(step_model_path)

    with codecs.open(step['base_config_path'], 'r', 'utf-8') as file:
        step_config = json.load(file)
    for key in special_tokens:
        step_config[key] = special_tokens[key]
    step_config['model_name'] = model_name_or_path

    step_config_path = os.path.join(step_model_path, 'step_config.json')
    with codecs.open(step_config_path, 'w', 'utf-8') as file:
        json.dump(step_config, file, ensure_ascii=False, indent=4)
    
    return step_model_path, step_config_path

def run_infer_model(model_path, output_dir, alpaca_eval_questions_path):
    output_path = os.path.join(output_dir, f'{os.path.basename(model_path)}_alpaca_eval.json')
    print(f'Infer {model_path} to {output_path}')
    call_res = subprocess.call(
        [
            'python', '-m', 'ruadapt.inference.infer_vllm', 
            model_path,
            alpaca_eval_questions_path,
            output_path,
            '--infer_for', 'alpaca_eval',
            '--max_samples', str(500)
        ]
    )
    if call_res:
        return call_res
    
    with codecs.open(output_path, 'r', 'utf-8') as file:
        data = json.load(file)

    for i in range(len(data)):
        data[i]['generator'] = os.path.basename(output_dir) + '_' + os.path.basename(model_path)
    
    with codecs.open(output_path, 'w', 'utf-8') as file:
        json.dump(data, file)

    return 0

def run_merge_model(lora_model_path):
    assert lora_model_path[-4:] == 'lora'
    print(f'Merging {lora_model_path} to {lora_model_path[:-5]}')
    return subprocess.call(
        [
            'python', 'scripts/merge_lora.py', 
            lora_model_path,
            lora_model_path[:-5]
        ]
    )

def run_lep(lep_model_path, lep_config_path, custom_chat_template_path):
    print(f'LEP to {lep_model_path}')
    return subprocess.call(
        [
            'python', '-m', 'ruadapt.ushanka.compose_ushanka', 
            '--config_path', lep_config_path,
            '--output_path', lep_model_path,
            '--mode', 'conversion',
            '--custom_chat_template_path', custom_chat_template_path
        ]
    )

def run_step(script, config_path, train_path, eval_path, output_path, sample_rate):
    print(f'Step {script} with {config_path} to {output_path} on {train_path}')
    return subprocess.call(
        [
            'python', '-m', script, 
            config_path,
            train_path,
            eval_path,
            output_path,
            str(sample_rate)
        ]
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ruadapt_base_model_name_or_path')
    parser.add_argument('--raw_base_model_name_or_path')
    parser.add_argument('--instruct_model_name_or_path')
    parser.add_argument('--custom_chat_template_path')
    parser.add_argument('--output_dir')
    parser.add_argument('--alpaca_eval_questions_path')
    parser.add_argument('--pipeline_config_path')
    parser.add_argument('--custom_bos_token', default=None)
    parser.add_argument('--custom_eos_token', default=None)
    parser.add_argument('--custom_pad_token', default=None)
    parser.add_argument('--sample_rate', type=float, default=1.0)
    parser.add_argument('--skip_lep', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        print(f'Creating dir {args.output_dir}')
        os.makedirs(args.output_dir)

    if args.skip_lep:
        print(f'WARNING: if tou skip lep step, then ruadapt_base_model_name_or_path ({args.ruadapt_base_model_name_or_path}) will be used as input instruct model for other steps!')
        prev_step_model_path = args.ruadapt_base_model_name_or_path

    if not args.skip_lep:
        lep_model_path, lep_config_path = create_lep_config(
            args.instruct_model_name_or_path, 
            args.raw_base_model_name_or_path, 
            args.ruadapt_base_model_name_or_path, 
            args.output_dir
        )

        if run_lep(lep_model_path, lep_config_path, args.custom_chat_template_path):
            print('ERROR while LEP. Stoping pipeline.')
            exit(1)
        
        if run_infer_model(lep_model_path, args.output_dir, args.alpaca_eval_questions_path):
            print('ERROR while infer LEP model. Stoping pipeline.')
            exit(1)
        prev_step_model_path = lep_model_path

    with codecs.open(args.pipeline_config_path, 'r', 'utf-8') as file:
        pipeline = json.load(file)

    special_tokens = resolve_special_tokens(
        prev_step_model_path, 
        args.custom_bos_token, 
        args.custom_eos_token, 
        args.custom_pad_token
    )
    
    for i, step in enumerate(pipeline):
        assert step['type'] in ['ft', 'sft', 'kto']
        step_model_path, step_config_path = create_step_config(i, step, prev_step_model_path, args.output_dir, special_tokens)
        if step['type'] == 'kto':
            engine = 'kto'
        else:
            engine = 'unsloth' if step['unsloth'] else 'transformers'
        script_name = f'ruadapt.instruct_tuning.train_{engine}'
        
        if run_step(script_name, step_config_path, step['train_file_path'], step['val_file_path'], step_model_path, args.sample_rate):
            print(f'ERROR while step {i}. Stoping pipeline.')
            exit(1)

        if run_merge_model(step_model_path):
            print(f'ERROR while step {i}. Stoping pipeline.')
            exit(1)

        prev_step_model_path = step_model_path[:-5]
        if run_infer_model(prev_step_model_path, args.output_dir, args.alpaca_eval_questions_path):
            print('ERROR while infer LEP model. Stoping pipeline.')
            exit(1)


        



        
        