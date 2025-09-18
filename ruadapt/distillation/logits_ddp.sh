#!/bin/bash
#SBATCH --job-name=logits_generation
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00                 
#SBATCH --output=/scratch/s02210660/ruadapt-test/distillation/logs/runs/logit_generation_%j.out     
#SBATCH --error=/scratch/s02210660/ruadapt-test/distillation/logs/runs/logit_generation_%j.err      

export NNODES=4
export GPUS_PER_NODE=8

export head_node=($(scontrol show hostnames $SLURM_JOB_NODELIST))
export head_node_ip=$(srun --nodes=1 --ntasks=1 --gpus 0 -w "$head_node" hostname --ip-address)

echo Head Node: $head_node
echo Head Node IP: $head_node_ip
echo "${head_node_ip: -1}"

export MODEL='RefalMachine/RuadaptQwen3-32B-Instruct'
export OUTPUT_PATH="distillation/${MODEL:13}_logits/teacher_train_maxlen8192_top100.parquet"
export DATA_PATH='/scratch/s02210660/ruadapt-test/ruadapt/hybrid_reasoning_dataset_ru_v4/train.jsonl'
export FILTERED_DATA="distillation/data/train_maxlen8192_top100.jsonl"
export MAXLEN=8192
export BS=1
export TOPK=100

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_BLOCKING_WAIT=1
#export TORCH_NCCL_ASYNC_ERROR_HANDLING=0



LOGGING_DIR="${MODEL:13}_logits/logs"

echo "Output directory: $OUTPUT_PATH"
echo "Logging directory: $LOGGING_DIR"

srun --container-image /scratch/s02210660/ruadapt-test/ngc_pytorch_25_05_training.sqsh \
     --container-workdir /scratch/s02210660/ruadapt-test/ \
     --container-mounts /scratch/s02210660/ruadapt-test/:/scratch/s02210660/ruadapt-test/ \
     bash -c "cd /scratch/s02210660/ruadapt-test && ./distillation/run_ddp_logits.sh \
                --model_name $MODEL \
                --input_file $DATA_PATH \
                --output_file $OUTPUT_PATH \
                --filtered_data_file $FILTERED_DATA \
                --max_length $MAXLEN \
                --batch_size $BS \
                --sort_by_len \
                --per_token \
                --topk $TOPK" 
