current_node=$(hostname)

echo Current Node: $current_node
echo Head Node Name: $head_node
echo Head Node IP: $head_node_ip

pip show pytest
pip install pytest==8.0.0
pip install transformers==4.45.2

echo $NNODES
echo $GPUS_PER_NODE
echo $HF_HOME
echo $RUADAPT_NO_TRAIN
rdzv_id="512${head_node_ip: -1}"
rdzv_port="2650${head_node_ip: -1}"
echo $rdzv_id
echo $rdzv_port
echo $MODEL_NAME_OR_PATH
echo $OUTPUT_DIR
echo $LR
echo $TRAIN_FILE_PATH

lora_rank=32
lora_alpha=32
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"
#lora_trainable="q_proj,v_proj,k_proj,o_proj"
#modules_to_save="embed_tokens"
modules_to_save="lm_head"
lora_dropout=0.05

#export HF_HOME=/scratch/tikhomirov/workdir/data/.cache/
torchrun --nnodes=$NNODES --nproc-per-node=$GPUS_PER_NODE --rdzv-id=$rdzv_id --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip:$rdzv_port train_trainer.py \
--model_name_or_path $MODEL_NAME_OR_PATH \
--train_file $TRAIN_FILE_PATH \
--validation_file /scratch/tikhomirov/workdir/data/darulm_20_05_24/val.json \
--block_size 4096 \
--preprocessing_num_workers 96 \
--output_dir $OUTPUT_DIR \
--overwrite_output_dir \
--do_train \
--do_eval \
--evaluation_strategy steps \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--learning_rate $LR \
--weight_decay 0.1 \
--adam_beta1 0.9 \
--adam_beta2 0.95 \
--adam_epsilon 1e-05 \
--num_train_epochs 1.0 \
--lr_scheduler_type cosine \
--warmup_steps 100 \
--save_step 10000 \
--logging_steps 10 \
--save_total_limit 8 \
--bf16 \
--bf16_full_eval \
--torch_dtype bfloat16 \
--gradient_accumulation_steps 4 \
--eval_steps 1000 \
--log_on_each_node false \
--peft $PEFT \
--lora_rank ${lora_rank} \
--lora_alpha ${lora_alpha} \
--trainable ${lora_trainable} \
--lora_dropout ${lora_dropout} \
--modules_to_save ${modules_to_save}

#--find_unused_parameters false
#--ddp_timeout 3600 \
#--deepspeed configs/deepspeed_z1_china.json
#--ddp_backend nccl \
#--ddp_bucket_cap_mb 100 \