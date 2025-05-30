from typing import Optional
import json

import fire
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import tiktoken
import shortuuid
from .utils import read_jsonl, read_json
import time
import os
import time

# taken from saiga project

def infer_vllm(
    model_name: str,
    input_path: str,
    output_path: str,
    temperature: float = 0.3,
    top_p: float = 0.9,
    top_k: int = 40,
    max_tokens: int = 2048,
    max_seq_len: int = 8192 // 2,
    repetition_penalty: float = 1.1,
    remove_bos_token: bool = False,
    quantization: Optional[str] = None,
    infer_for: str = 'default',
    max_samples: int = -1,
    add_no_think=False
):
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
    )
    llm = LLM(
        model=model_name,
        max_seq_len_to_capture=max_seq_len,
        max_model_len=max_seq_len,
        quantization=quantization,
    )
    tokenizer = llm.get_tokenizer()
    try:
        records = read_json(input_path)
    except:
        records = read_jsonl(input_path)
    role_mapping = {
        "bot": "assistant",
        "gpt": "assistant",
        "human": "user",
    }
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    prompts = []
    #if max_samples > 0:
    #    records = records[:max_samples]

    records_filtered = []
    record_key = ''
    for r in records:
        if "instruction" in r:
            messages = [{"role": "user", "content": r["instruction"]}]
            record_key = 'instruction'
        elif "messages" in r:
            messages = r.get("messages", r.get("prompt", r.get('turns')))
            record_key = 'messages'
        elif "prompt" in r or 'turns' in r:
            messages = r.get("messages", r.get("prompt", r.get('turns')))
            record_key = 'prompt'
        elif 'turns' in r:
            messages = r.get("messages", r.get("prompt", r.get('turns')))
            record_key = 'turns'

        assert messages
        for i in range(len(messages)):
            if 'role' not in messages[i]:
                assert len(messages) == 1
                messages[i]['role'] = 'user'

        if len([m for m in messages if m['role'] in ['assistant', 'bot']]) > 1:
            continue

        for m in messages:
            m["role"] = role_mapping.get(m["role"], m["role"])
        if messages[-1]["role"] == "assistant":
            messages = messages[:-1]

        #if add_no_think:
        #    messages[-1]['content'] += ' /no_think'
        if add_no_think:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
            )
        else:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        #if remove_bos_token:
        #    prompt = prompt.replace(tokenizer.bos_token, "")
        if len(prompt) < 2000:
            prompts.append(prompt)
            records_filtered.append(r)

    records = records_filtered

    if max_samples > 0:
        records = records[:max_samples]
        prompts = prompts[:max_samples]
    print(len(records))
    print(prompts[0])
    print(sampling_params)
    s = time.time()
    outputs = llm.generate(prompt_token_ids=prompts, sampling_params=sampling_params)
    full_results = []
    gen_text_len = []
    with open(output_path, "w") as w:
        j = 0
        for record, output in zip(records, outputs):
            prompt_token_ids = output.prompt_token_ids
            prompt = tokenizer.decode(output.prompt_token_ids)
            assert prompt_token_ids[0] != prompt_token_ids[1], prompt_token_ids
            generated_text = output.outputs[0].text

            choices = []
            turns = []
            turns.append({"content": generated_text, "token_len": len(encoding.encode(generated_text, allowed_special=set({'<|endoftext|>'})))})
            gen_text_len.append(len(encoding.encode(generated_text, allowed_special=set({'<|endoftext|>'}))))
            choices.append({"index": i, "turns": turns})
            ans = {
                "question_id": record.get("question_id", record[record_key]),
                "answer_id": shortuuid.uuid(),
                "model_id": os.path.basename(model_name),
                "choices": choices,
                "tstamp": time.time(),
            }
            j += 1

            print(prompt)
            print(generated_text)
            print(prompt_token_ids)
            print()
            print()
            record["answer"] = generated_text.encode("utf-8").decode("utf-8", "ignore")
            if infer_for == 'default':
                w.write(json.dumps(ans, ensure_ascii=False).strip() + "\n")
            elif infer_for == 'alpaca_eval':
                sample = {
                    "instruction": record['instruction'],
                    "output": record["answer"],
                    "generator": os.path.basename(model_name),
                    "dataset": os.path.basename(input_path),
                    "datasplit": "eval"
                }
                full_results.append(sample)
            else:
                raise Exception('ERROR: infer_for')
        if infer_for == 'alpaca_eval':
            json.dump(full_results, w, ensure_ascii=False, indent=4)
    print('AVERAGE_LEN: ' + str(sum(gen_text_len) / len(gen_text_len)))
    total_time = time.time() - s
    print(total_time)
    print((sum(gen_text_len) / len(gen_text_len)) / total_time)

if __name__ == "__main__":
    fire.Fire(infer_vllm)

