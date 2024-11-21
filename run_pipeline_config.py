import argparse
import os
import subprocess
import shutil
import codecs
import json
from transformers import AutoTokenizer

def hash(f, args, kwargs):
    f_name = f.__name__
    args_str = '__'.join(list(map(str, args)))
    kwargs_sorted = sorted(kwargs.items(), key=lambda x: x[0])
    kwargs_sorted_str = '__'.join(['::'.join(list(map(str, d))) for d in kwargs_sorted])
    return '|'.join([f_name, args_str, kwargs_sorted_str])

HASH_PATH = None
def check_op(func):
    def inner1(*args, **kwargs):
        assert HASH_PATH is not None
        with codecs.open(HASH_PATH, 'r', 'utf-8') as file:
            info = json.load(file)
        op_hash = hash(func, args, kwargs)
        if op_hash in info:
            return info[op_hash]
        
        returned_value = func(*args, **kwargs)
        if returned_value:
            return returned_value
        info[op_hash] = returned_value
        with codecs.open(HASH_PATH, 'w', 'utf-8') as file:
            json.dump(info, file)

        return returned_value
        
    return inner1

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

def create_lep_config(target_model_path, source_model_path, donor_model_path, output_dir, alpha_scale=1.0, not_scale_lm_head=False):
    lep_model_path = os.path.join(output_dir, 'lep')
    if not os.path.exists(lep_model_path):
        os.mkdir(lep_model_path)

    lep_config = {
        "target_model_path" : target_model_path,
        "source_model_path" : source_model_path,
        "donor_model_path" : donor_model_path,
        "alpha_scale": alpha_scale,
        "not_scale_lm_head": not_scale_lm_head
    }

    lep_config_path = os.path.join(lep_model_path, 'lep_config.json')
    with codecs.open(lep_config_path, 'w', 'utf-8') as file:
        json.dump(lep_config, file, ensure_ascii=False, indent=4)

    return lep_model_path, lep_config_path

def create_step_config(step_idx, step, model_name_or_path, output_dir, special_tokens, num_gpu=1):
    step_model_path = os.path.join(output_dir, step['type'] + str(step_idx) + '_lora')
    if not os.path.exists(step_model_path):
        os.mkdir(step_model_path)

    with codecs.open(step['base_config_path'], 'r', 'utf-8') as file:
        step_config = json.load(file)
    print(step_config)
    for key in special_tokens:
        step_config[key] = special_tokens[key]
    step_config['model_name'] = model_name_or_path

    if num_gpu > 1:
        step_config['trainer']['gradient_accumulation_steps'] = max(1, int(step_config['trainer']['gradient_accumulation_steps'] / num_gpu))

    step_config_path = os.path.join(step_model_path, 'step_config.json')
    with codecs.open(step_config_path, 'w', 'utf-8') as file:
        json.dump(step_config, file, ensure_ascii=False, indent=4)
    
    return step_model_path, step_config_path

@check_op
def run_infer_model(model_path, output_dir, alpaca_eval_questions_path):
    output_path = os.path.join(output_dir, f'{os.path.basename(model_path)}_alpaca_eval.json')
    print(f'Infer {model_path} to {output_path}')
    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = "0"
    call_res = subprocess.call(
        [
            'python', '-m', 'ruadapt.inference.infer_vllm', 
            model_path,
            alpaca_eval_questions_path,
            output_path,
            '--infer_for', 'alpaca_eval',
            '--max_samples', str(500)
        ], env=my_env
    )
    if call_res:
        return call_res
    
    with codecs.open(output_path, 'r', 'utf-8') as file:
        data = json.load(file)

    for i in range(len(data)):
        data[i]['generator'] = os.path.basename(output_dir) + '_' + os.path.basename(model_path)
    
    with codecs.open(output_path, 'w', 'utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    return 0

@check_op
def run_merge_model(lora_model_path, alpha_scale=1.0):
    assert lora_model_path[-4:] == 'lora'
    print(f'Merging {lora_model_path} to {lora_model_path[:-5]} with as={alpha_scale}')
    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = "0"
    return subprocess.call(
        [
            'python', 'scripts/merge_lora.py', 
            lora_model_path,
            lora_model_path[:-5],
            '--alpha_scale', str(alpha_scale)
        ], env=my_env
    )

@check_op
def run_lep(lep_model_path, lep_config_path, custom_chat_template_path):
    print(f'LEP to {lep_model_path}')
    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = "0,1"
    return subprocess.call(
        [
            'python', '-m', 'ruadapt.ushanka.compose_ushanka', 
            '--config_path', lep_config_path,
            '--output_path', lep_model_path,
            '--mode', 'conversion',
            '--custom_chat_template_path', custom_chat_template_path
        ], env=my_env
    )

@check_op
def run_step(script, config_path, train_path, eval_path, output_path, custom_chat_template_path, sample_rate, num_gpu=1):
    print(f'Step {script} with {config_path} to {output_path} on {train_path} with {num_gpu}')
    if num_gpu == 1:
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = "0"
        return subprocess.call(
            [
                'python', '-m', script, 
                config_path,
                train_path,
                eval_path,
                output_path,
                custom_chat_template_path,
                str(sample_rate)
            ], env=my_env
        )
    else:
        #my_env = os.environ.copy()
        #my_env["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in range(num_gpu)])
        return subprocess.call(
            [
                'torchrun',
                '--nnodes', str(1),
                '--nproc-per-node', str(num_gpu),
                '-m', script, 
                config_path,
                train_path,
                eval_path,
                output_path,
                custom_chat_template_path,
                str(sample_rate)
            ]#, env=my_env
        )

@check_op
def eval_instruct_model_zero_shot(model_name_or_path, output_dir=None, num_gpu=1):
    if output_dir is None:
        output_dir = os.path.join(model_name_or_path, 'llmtf_eval')
    
    if 'qwen' in model_name_or_path:
        conv_path = 'conversation_configs/qwen2.json'
    else:
        conv_path = 'conversation_configs/llama3_no_system.json'

    print(f'Eval {model_name_or_path} to {output_dir} with {conv_path}')
    return subprocess.call(
        [
            'torchrun',
            '--nnodes', str(1),
            '--nproc-per-node', str(num_gpu),
            'run_evaluate_multinode_multigpu.py',
            '--conv_path', conv_path,
            '--model_dir', model_name_or_path,
            '--output_dir', output_dir,
            '--batch_size', str(2),
            '--max_len', str(4000),
            '--few_shot_count', str(0),
            '--short'
        ], cwd='./ruadapt/evaluation/llmtf_open'
     )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path')
    args = parser.parse_args()

    print('HELLO THERE')
    with codecs.open(args.config_path, 'r', 'utf-8') as file:
        params = json.load(file)
        for key in params:
            args.__setattr__(key, params[key])
    print(args)
    
    if not os.path.exists(args.output_dir):
        print(f'Creating dir {args.output_dir}')
        os.makedirs(args.output_dir)

    HASH_PATH = os.path.join(args.output_dir, '.hash.json')
    if not os.path.exists(HASH_PATH):
        with codecs.open(HASH_PATH, 'w', 'utf-8') as file:
            json.dump({}, file)

    if args.skip_lep:
        print(f'WARNING: if tou skip lep step, then ruadapt_base_model_name_or_path ({args.ruadapt_base_model_name_or_path}) will be used as input instruct model for other steps!')
        prev_step_model_path = args.ruadapt_base_model_name_or_path
    else:
        prev_step_model_path = args.instruct_model_name_or_path

    #if args.eval:
    #    if eval_instruct_model_zero_shot(prev_step_model_path, os.path.join(args.output_dir, 'init_llmtf_eval'), num_gpu=args.num_gpu):
    #        print(f'ERROR: failed to eval {prev_step_model_path}, but continue')

    if not args.skip_lep:
        lep_model_path, lep_config_path = create_lep_config(
            args.instruct_model_name_or_path, 
            args.raw_base_model_name_or_path, 
            args.ruadapt_base_model_name_or_path, 
            args.output_dir,
            args.alpha_scale,
            args.not_scale_lm_head
        )

        if run_lep(lep_model_path, lep_config_path, args.custom_chat_template_path):
            print('ERROR while LEP. Stoping pipeline.')
            exit(1)
        
        #if run_infer_model(lep_model_path, args.output_dir, args.alpaca_eval_questions_path):
        #    print('ERROR while infer LEP model. Stoping pipeline.')
        #    exit(1)

        prev_step_model_path = lep_model_path
        if args.eval:
            if eval_instruct_model_zero_shot(prev_step_model_path, num_gpu=args.num_gpu):
                print(f'ERROR: failed to eval {prev_step_model_path}, but continue')

    with codecs.open(args.pipeline_config_path, 'r', 'utf-8') as file:
        pipeline = json.load(file)

    shutil.copyfile(args.pipeline_config_path, os.path.join(args.output_dir, 'pipeline_config.json'))

    special_tokens = resolve_special_tokens(
        prev_step_model_path, 
        args.custom_bos_token, 
        args.custom_eos_token, 
        args.custom_pad_token
    )
    
    for i, step in enumerate(pipeline):
        assert step['type'] in ['ft', 'sft', 'kto', 'simpo']
        step_model_path, step_config_path = create_step_config(i, step, prev_step_model_path, args.output_dir, special_tokens, num_gpu=args.num_gpu)
        if step['type'] == 'kto':
            engine = 'kto'
        elif step['type'] == 'simpo':
            engine = 'cpo'
        else:
            engine = 'sft'
        if not step['unsloth']:
            engine += '_transformers'
        else:
            assert args.num_gpu == 1

        script_name = f'ruadapt.instruct_tuning.train_{engine}'

        if run_step(script_name, step_config_path, step['train_file_path'], step['val_file_path'], step_model_path, args.custom_chat_template_path, args.sample_rate, num_gpu=args.num_gpu):
            print(f'ERROR while step {i}. Stoping pipeline.')
            exit(1)

        if run_merge_model(step_model_path, step.get('alpha_scale', 1.0)):
            print(f'ERROR while step {i}. Stoping pipeline.')
            exit(1)

        prev_step_model_path = step_model_path[:-5]
        #if run_infer_model(prev_step_model_path, args.output_dir, args.alpaca_eval_questions_path):
        #    print(f'ERROR while infer step{i} model. Stoping pipeline.')
        #    exit(1)

        if args.eval:
            if eval_instruct_model_zero_shot(prev_step_model_path, num_gpu=args.num_gpu):
                print(f'ERROR: failed to eval {prev_step_model_path}, but continue')