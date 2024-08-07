#!/bin/bash
seed=$1   # desired seed
algo=$2   # ardt, dt, esper
device=$3 # cpu, cuda

python main.py \
    --seed $seed \
    --data_name "mstoy" \
    --env_name "mstoy" \
    --ret_file "data/${algo}_test_ll${ll}" \
    --device $device \
    --algo $algo \
    --config "configs/${algo}/test.yaml" \
    --checkpoint_dir "checkpoints/${algo}_test_seed${seed}" \
    --model_type "dt" \
    --K 4 \
    --max_iters 1 \
    --num_steps_per_iter 10
