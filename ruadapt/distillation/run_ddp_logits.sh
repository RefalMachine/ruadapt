#!/bin/bash
current_node=$(hostname)

echo Current Node: $current_node
echo Head Node Name: $head_node
echo Head Node IP: $head_node_ip

rdzv_id="512${head_node_ip: -1}"
rdzv_port="2650${head_node_ip: -1}"
echo $rdzv_id
echo $rdzv_port 

torchrun --nnodes=$NNODES \
         --nproc-per-node=$GPUS_PER_NODE \
         --rdzv-id=$rdzv_id \
         --rdzv-backend=c10d \
         --rdzv-endpoint=${head_node_ip}:${rdzv_port} \
         -m distillation.logits_ddp "$@" 