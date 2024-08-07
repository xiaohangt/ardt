#!/bin/bash
seed=$1                 # desired seed
algo=$2                 # ardt, dt, esper
device=$3               # cpu, cuda
ll=$4                   # leaf weight, range [0, 1], recommended 0.9
alpha=$5                # alpha, range [0, 1], recommended 0.01
model_type=$6           # adt, dt, bc
env_name=$7             # halfcheetah, hopper, walker2d
mix_coef=$8             # proportion of random data, range [0, 1]
env_alpha=$9            # alpha for action weights, range [0, 1], default 0.1
num_eval_episodes=${10} # number of evaluation episodes, default 100

d_name="arrl_train_${env_name}_high"
added_data_name="random_${env_name}"
adv_base="arrl/models/"
adv="${adv_base}${env_name}/ou_noise/nr_mdp_0.1_1/0"

# mix_coef here defines the proportion between collected and random data
for added_data_prop in "${mix_coef}"
do
    ret_file="data/${algo}_${d_name}_${added_data_name}_${added_data_prop}"

    # Training
    python main.py \
        --seed $seed \
        --data_name $d_name \
        --added_data_name $added_data_name \
        --added_data_prop $added_data_prop \
        --env_name $env_name \
        --ret_file $ret_file \
        --device $device \
        --algo $algo \
        --config "configs/${algo}/mujoco.yaml" \
        --batch_size 512 \
        --leaf_weight $ll \
        --alpha $alpha \
        --checkpoint_dir "checkpoints/${algo}_${model_type}_${d_name}_${added_data_name}_${added_data_prop}_seed${seed}" \
        --K 20 \
        --model_type $model_type \
        --max_iters 12 \
        --num_steps_per_iter 100000 \
        --test_adv $adv \
        --env_alpha $env_alpha \
        --num_eval_episodes 0

    # Testing
    python main.py \
        --is_training \
        --seed $seed \
        --data_name $d_name \
        --added_data_name $added_data_name \
        --added_data_prop $added_data_prop \
        --env_name $env_name \
        --ret_file $ret_file \
        --device $device \
        --algo $algo \
        --config "configs/${algo}/mujoco.yaml" \
        --batch_size 512 \
        --leaf_weight $ll \
        --alpha $alpha \
        --checkpoint_dir "checkpoints/${algo}_${model_type}_${d_name}_${added_data_name}_${added_data_prop}_seed$(( $seed % 5 ))" \
        --K 20 \
        --model_type $model_type \
        --max_iters 0 \
        --num_steps_per_iter 0 \
        --test_adv $adv \
        --env_alpha $env_alpha \
        --num_eval_episodes $num_eval_episodes
done
