#!/bin/bash
seed=$1                 # desired seed
algo=$2                 # ardt, dt, esper
device=$3               # cpu, cuda
model_type=$4           # adt, dt, bc
env_name=$5             # halfcheetah, hopper, walker2d
mix_coef=$6             # proportion of random data, range [0, 1]
env_alpha=$7            # alpha for action weights, range [0, 1], default 0.1
num_eval_episodes=$8    # number of evaluation episodes, default 100

d_name="arrl_train_${env_name}_high"
added_data_name="random_${env_name}"
adv_base="arrl/models/"
adv="${adv_base}${env_name}/ou_noise/nr_mdp_0.1_1/0"

# mix_coef here defines the proportion between collected and random data
for added_data_prop in "${mix_coef}"
do
    ret_file="offline_data/${algo}_${d_name}_${added_data_name}_${added_data_prop}"

    # Training
    python main.py \
        --is_training_only \
        --seed $seed \
        --data_name $d_name \
        --added_data_name $added_data_name \
        --added_data_prop $added_data_prop \
        --env_name $env_name \
        --ret_file $ret_file \
        --device $device \
        --algo $algo \
        --config "configs/${algo}/mujoco.yaml" \
        --checkpoint_dir "checkpoints/${algo}_${model_type}_${d_name}_${added_data_name}_${added_data_prop}_seed${seed}" \
        --K 20 \
        --model_type $model_type \
        --train_iters 12 \
        --num_steps_per_iter 100000 \
        --test_adv $adv \
        --env_alpha $env_alpha \
        --num_eval_episodes 0

    # Testing
    python main.py \
        --is_testing_only \
        --seed $seed \
        --data_name $d_name \
        --added_data_name $added_data_name \
        --added_data_prop $added_data_prop \
        --env_name $env_name \
        --ret_file $ret_file \
        --device $device \
        --algo $algo \
        --config "configs/${algo}/mujoco.yaml" \
        --checkpoint_dir "checkpoints/${algo}_${model_type}_${d_name}_${added_data_name}_${added_data_prop}_seed$(( $seed % 5 ))" \
        --K 20 \
        --model_type $model_type \
        --train_iters 0 \
        --num_steps_per_iter 0 \
        --test_adv $adv \
        --env_alpha $env_alpha \
        --num_eval_episodes $num_eval_episodes
done
