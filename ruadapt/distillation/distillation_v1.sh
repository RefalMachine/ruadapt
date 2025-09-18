#!/bin/bash
#SBATCH --job-name=qwen_distillation
#SBATCH --nodes=4
#SBATCH --gres=gpu:8             
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00                 
#SBATCH --output=/scratch/s02210660/ruadapt-test/distillation/logs/runs/distillation_%j.out     
#SBATCH --error=/scratch/s02210660/ruadapt-test/distillation/logs/runs/distillation_%j.err      

export NNODES=4
export GPUS_PER_NODE=8

export head_node=($(scontrol show hostnames $SLURM_JOB_NODELIST))
export head_node_ip=$(srun --nodes=1 --ntasks=1 --gpus 0 -w "$head_node" hostname --ip-address)

echo Head Node: $head_node
echo Head Node IP: $head_node_ip
echo "${head_node_ip: -1}"

export OUTPUT_DIR=/scratch/s02210660/ruadapt-test/distillation/distillation_experiments/distill_r128_a256_lr1e-4_top100_m1/distill_lora
export STEP_CONFIG_PATH=$OUTPUT_DIR/step_config.json
echo "Output directory: $OUTPUT_DIR"

srun --container-image /scratch/s02210660/ruadapt-test/ngc_pytorch_25_05_training.sqsh \
     --container-workdir /scratch/s02210660/ruadapt-test \
     --container-mounts /scratch/s02210660/ruadapt-test:/scratch/s02210660/ruadapt-test \
     bash -c "cd /scratch/s02210660/ruadapt-test/distillation && ./run_distill_sft.sh \
                --config_file $STEP_CONFIG_PATH \
                --output_dir $OUTPUT_DIR"

export MODEL_TO_EVAL=${OUTPUT_DIR:0:-5}
srun --nodes=1 --gpus 1 --container-image /scratch/s02210660/ruadapt-test/ngc_pytorch_25_05_training.sqsh \
     --container-workdir /scratch/s02210660 \
     --container-mounts /scratch/s02210660:/scratch/s02210660 bash -c "cd /scratch/s02210660/ruadapt-test/ruadapt/scripts && python merge_lora.py $OUTPUT_DIR $MODEL_TO_EVAL"

AS=1.25
export MODEL_TO_EVAL=${OUTPUT_DIR:0:-5}"_as_1.25"
srun --nodes=1 --gpus 1 --container-image /scratch/s02210660/ruadapt-test/ngc_pytorch_25_05_training.sqsh \
     --container-workdir /scratch/s02210660 \
     --container-mounts /scratch/s02210660:/scratch/s02210660 bash -c "cd /scratch/s02210660/ruadapt-test/ruadapt/scripts && python merge_lora.py $OUTPUT_DIR $MODEL_TO_EVAL cuda $AS" &

AS=1.5
export MODEL_TO_EVAL=${OUTPUT_DIR:0:-5}"_as_1.5"
srun --nodes=1 --gpus 1 --container-image /scratch/s02210660/ruadapt-test/ngc_pytorch_25_05_training.sqsh \
     --container-workdir /scratch/s02210660 \
     --container-mounts /scratch/s02210660:/scratch/s02210660 bash -c "cd /scratch/s02210660/ruadapt-test/ruadapt/scripts && python merge_lora.py $OUTPUT_DIR $MODEL_TO_EVAL cuda $AS" &

AS=1.75
export MODEL_TO_EVAL=${OUTPUT_DIR:0:-5}"_as_1.75"
srun --nodes=1 --gpus 1 --container-image /scratch/s02210660/ruadapt-test/ngc_pytorch_25_05_training.sqsh \
     --container-workdir /scratch/s02210660 \
     --container-mounts /scratch/s02210660:/scratch/s02210660 bash -c "cd /scratch/s02210660/ruadapt-test/ruadapt/scripts && python merge_lora.py $OUTPUT_DIR $MODEL_TO_EVAL cuda $AS" &

AS=2
export MODEL_TO_EVAL=${OUTPUT_DIR:0:-5}"_as_2"
srun --nodes=1 --gpus 1 --container-image /scratch/s02210660/ruadapt-test/ngc_pytorch_25_05_training.sqsh \
     --container-workdir /scratch/s02210660 \
     --container-mounts /scratch/s02210660:/scratch/s02210660 bash -c "cd /scratch/s02210660/ruadapt-test/ruadapt/scripts && python merge_lora.py $OUTPUT_DIR $MODEL_TO_EVAL cuda $AS" 

wait
