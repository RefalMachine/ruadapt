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
    max_samples: int = -1
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
        quantization=quantization,
    )
    tokenizer = llm.get_tokenizer()
    records = read_json(input_path)
    role_mapping = {
        "bot": "assistant",
        "gpt": "assistant",
        "human": "user",
    }
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    prompts = []
    if max_samples > 0:
        records = records[:max_samples]
    for r in records:
        if "instruction" in r:
            messages = [{"role": "user", "content": r["instruction"]}]
        elif "messages" in r or "prompt" in r or 'turns' in r:
            messages = r.get("messages", r.get("prompt", r.get('turns')))

        assert messages
        for i in range(len(messages)):
            if 'role' not in messages[i]:
                assert len(messages) == 1
                messages[i]['role'] = 'user'

        for m in messages:
            m["role"] = role_mapping.get(m["role"], m["role"])
        if messages[-1]["role"] == "assistant":
            messages = messages[:-1]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        #if remove_bos_token:
        #    prompt = prompt.replace(tokenizer.bos_token, "")

        prompts.append(prompt)

    print(prompts[0])
    print(sampling_params)
    outputs = llm.generate(prompt_token_ids=prompts, sampling_params=sampling_params)
    full_results = []
    with open(output_path, "w") as w:
        for record, output in zip(records, outputs):
            prompt_token_ids = output.prompt_token_ids
            prompt = tokenizer.decode(output.prompt_token_ids)
            assert prompt_token_ids[0] != prompt_token_ids[1], prompt_token_ids
            generated_text = output.outputs[0].text

            choices = []
            turns = []
            turns.append({"content": generated_text, "token_len": len(encoding.encode(generated_text))})
            choices.append({"index": i, "turns": turns})
            ans = {
                "question_id": record.get("question_id", record['instruction']),
                "answer_id": shortuuid.uuid(),
                "model_id": os.path.basename(model_name),
                "choices": choices,
                "tstamp": time.time(),
            }

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

if __name__ == "__main__":
    fire.Fire(infer_vllm)

