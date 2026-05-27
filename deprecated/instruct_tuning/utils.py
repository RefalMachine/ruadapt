import json
import os
import random
import torch
import numpy as np
from tqdm import tqdm 
import re

def read_jsonl(file_name):
    with open(file_name, encoding="utf-8") as r:
        data = [l for l in r]
    for i in tqdm(range(len(data))):
        try:
            data[i] = json.loads(data[i])
        except:
            print(data[i])
            data[i] = ''
            continue
    print(len(data))
    print(len([d for d in data if len(d) > 0]))
    return [d for d in data if len(d) > 0]

def read_json(file_name):
    with open(file_name, encoding="utf-8") as r:
        return json.load(r)
    
def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as w:
        for r in records:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def gen_batch(records, batch_size):
    batch_start = 0
    while batch_start < len(records):
        batch_end = batch_start + batch_size
        batch = records[batch_start: batch_end]
        batch_start = batch_end
        yield batch

def prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    use_reentrant=True,
):

    # Freeze all parameters except LoRA
    with torch.no_grad():
        for name, param in model.named_parameters():
            if ".lora_A." in name or ".lora_B." in name or ".lora_magnitude_vector" in name:
                param.requires_grad_(True)
                # Also must be in float32!
                if param.dtype != torch.float32:
                    name = name.replace("base_model", "model", 1)
                    layer_number = re.search(r"\.[\d]{1,}\.", name).group(0)
                    name = name.replace(layer_number, f"[{layer_number[1:-1]}].")
                    name = name.replace(".weight", "", 1)
                    exec(f"{name}.to(torch.float32)")
                pass
            else:
                param.requires_grad_(False)
        pass
    pass

    if use_gradient_checkpointing == True:
        model.gradient_checkpointing_enable({"use_reentrant": use_reentrant})
    pass

    # If use_reentrant = True which is the Pytorch default, we just make the input requires_grad.
    if use_reentrant:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    return model