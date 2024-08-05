#!/bin/bash
seed=$1
algo=$2      # ardt, dt, esper
env_name=$3
device=$4
mix_coef=$5

model_type=$6
ll=0.9
env_alpha=0.1

d_name="arrl_train_${env_name}_high"
added_data_name="random_${env_name}"
adv_base=../arrl/models/
adv="${adv_base}${env_name}/ou_noise/nr_mdp_0.1_1/0"

# mix_coef here defines the proportion between collected and random data
for added_data_prop in "${mix_coef}"
do
    # Training
    ret_file="../data/${algo}_${d_name}_${added_data_name}_${added_data_prop}"

    python ../main.py \
        --ret_file $ret_file \
        --added_data_name $added_data_name \
        --added_data_prop $added_data_prop \
        --model_type $model_type \
        --n_cpu 1 \
        --seed $seed \
        --algo $algo \
        --device "cuda:${device}" \
        --data_name $d_name \
        --env_name $env_name \
        --test_adv $adv \
        --config "../configs/${algo}/halfcheetah.yaml" \
        --max_iters 1 \
        --num_steps_per_iter 100000 \
        --num_eval_episodes 0 \
        --K 20 \
        --checkpoint_dir "../checkpoints/${algo}_${model_type}_${d_name}_${added_data_name}_${added_data_prop}_seed${seed}"


    # Testing
    python ../main.py \
        --is_training \
        --env_alpha $env_alpha \
        --added_data_prop $added_data_prop \
        --model_type $model_type \
        --added_data_name $added_data_name \
        --seed $seed \
        --algo $algo \
        --device "cuda:${device}" \
        --data_name $d_name \
        --env_name $env_name \
        --test_adv $adv \
        --config "../configs/${algo}/halfcheetah.yaml" \
        --ret_file $ret_file \
        --max_iters 0 \
        --num_steps_per_iter 0 \
        --K 20 \
        --checkpoint_dir "../checkpoints/${algo}_${model_type}_${d_name}_${added_data_name}_${added_data_prop}_seed$(( $seed % 5 ))"
        
done
