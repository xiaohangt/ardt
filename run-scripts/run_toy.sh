#!/bin/bash
seed=$1
algo=$2       # ardt, dt, esper

env_name="toy"
device=0

model_type=dt
ll=0.9

python main.py \
    --model_type $model_type \
    --device "cuda:${device}" \
    --data_name $env_name \
    --env_name $env_name \
    --algo $algo \
    --seed $seed \
    --config "../configs/${algo}/${env_name}.yaml" \
    --ret_file "../data/${algo}_${env_name}_new_ll${ll}" \
    --max_iters 1 \
    --num_steps_per_iter 10000 \
    --K 4 \
    --checkpoint_dir "../checkpoints/radt_${d_name}_seed${seed}" 
