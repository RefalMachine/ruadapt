from fastapi import FastAPI
import uvicorn
from anyio.lowlevel import RunVar
from anyio import CapacityLimiter
from fastapi import Request
import threading
import time
import subprocess
import json
import codecs
from fastapi import FastAPI, HTTPException, Request, Response, Query, Depends
from pydantic import BaseModel, Field
import os

app = FastAPI()

class Params(BaseModel):
    ruadapt_base_model_name_or_path: str = Field(
        Query(
            min_length=0, max_length=5000
        )
    )
    raw_base_model_name_or_path: str = Field(
        Query(
            min_length=0, max_length=5000,
            default=''
        )
    )
    instruct_model_name_or_path: str = Field(
        Query(
            min_length=0, max_length=5000,
            default=''
        )
    )
    output_dir: str = Field(
        Query(
            min_length=1, max_length=5000
        )
    )
    pipeline_config_path: str = Field(
        Query(
            min_length=1, max_length=5000
        )
    )
    alpaca_eval_questions_path: str = Field(
        Query(
            min_length=1, max_length=5000
        )
    )

    custom_bos_token: str | None = Field(
        Query(
            min_length=0, max_length=100, 
            default=None
        )
    )
    custom_eos_token: str | None = Field(
        Query(
            min_length=0, max_length=100, 
            default=None
        )
    )
    custom_pad_token: str | None = Field(
        Query(
            min_length=0, max_length=100, 
            default=None
        )
    )
    skip_lep: bool = Field( 
        Query(
            default=False
        )
    )
    eval: bool = Field( 
        Query(
            default=True
        )
    )
    sample_rate: float = Field( 
        Query(
            ge=0.0, le=1.0, 
            default=1.0
        )
    )

q = []
@app.on_event("startup")
def startup():
    print("start")
    RunVar("_default_thread_limiter").set(CapacityLimiter(1))

@app.get('/get')
def get():
    return q

@app.get('/add')
def add(query: Params = Depends()):
    q.append(query)

devices = ['cuda1', 'cuda2', 'cuda3', 'cuda4']
def check_available_devices():
    res = subprocess.check_output(['docker', 'ps']).decode('utf-8')
    res = [r.split()[-1] for r in res.split('\n') if len(r.strip()) > 0]
    available_devices = [d for d in devices if d not in res]
    return available_devices

def query2params(query):
    params = query.dict()
    return params

def start_pipeline(params, device):
    print('Start experiment with:')
    print(params)
    print(f'on device {device}')
    with codecs.open('tmp_config.json', 'w', 'utf-8') as file:
        json.dump(params, file, ensure_ascii=False, indent=4)

    call = 'python run_pipeline_config.py --config_path tmp_config.json'
    wandb_api_key = os.environ['WANDB_API_KEY']
    print(wandb_api_key)
    full_call = f'docker run -v /home/maindev:/workdir -it --gpus \'\"device={device[-1]}\"\' --rm -d --name {device} ngc_cuda_pytorch_vllm_11_10_24_v7 bash -c \"cd projects/ruadapt && WANDB_API_KEY={wandb_api_key} {call}\"'
    print(full_call)
    print(subprocess.call(
        full_call, shell=True
    ))

def loop():
    while True:
        time.sleep(5)
        available_devices = check_available_devices()
        if len(available_devices) > 0 and len(q) > 0:
            query = q.pop(0)
            params = query2params(query)
            threading.Thread(target=start_pipeline, args=[params, available_devices[0]]).start()

t = threading.Thread(target=loop)
t.start()

uvicorn.run(app, host='0.0.0.0', port=8107, workers=1)

