echo Node IP: $head_node_ip
echo Node name: $head_node

lora_rank=16
lora_alpha=128
lora_trainable="lm_head,q_proj,v_proj,k_proj,o_proj"
modules_to_save="embed_tokens"
lora_dropout=0.05

torchrun --nnodes=8 --nproc-per-node=2 --rdzv-id=5123 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip:26423 train_trainer_peft_raw_load.py \
--model_name_or_path outputs/solar_10.7_darulm_unigram_proj_init_8node_darulm_part1_v3_1.0_512_12_02_24 \
--train_file ../darulm_12_02_24_part1/train_0.2_part2.json \
--validation_file ../darulm_12_02_24_part1/val.json \
--block_size 512 \
--preprocessing_num_workers 12 \
--output_dir outputs/solar_10.7_darulm_unigram_proj_init_8node_darulm_part2_v3_1.0_512_12_02_24 \
--overwrite_output_dir \
--do_train \
--do_eval \
--evaluation_strategy steps \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--learning_rate 5e-05 \
--weight_decay 0.1 \
--adam_beta1 0.9 \
--adam_beta2 0.95 \
--adam_epsilon 1e-05 \
--num_train_epochs 1.0 \
--lr_scheduler_type linear \
--warmup_steps 0 \
--logging_steps 50 \
--save_steps 1000 \
--save_total_limit 40 \
--fp16_full_eval \
--fp16 \
--fp16_opt_level O3 \
--torch_dtype float16 \
--gradient_accumulation_steps 8 \
--eval_steps 500 \
--ddp_timeout 3600 \
--log_on_each_node false \
--deepspeed configs/deepspeed_llama_z1_china.json \
--lora_rank ${lora_rank} \
--lora_alpha ${lora_alpha} \
--trainable ${lora_trainable} \
--lora_dropout ${lora_dropout} \
--modules_to_save ${modules_to_save}
