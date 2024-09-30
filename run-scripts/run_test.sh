#!/bin/bash
seed=$1                 # desired seed
algo=$2                 # ardt, dt, esper
device=$3               # cpu, cuda
model_type=$4           # adt, dt, bc
env_name=$5             # halfcheetah, hopper, walker2d
mix_coef=$6             # proportion of random data, range [0, 1]
env_alpha=$7            # alpha for action weights, range [0, 1], default 0.1
num_eval_episodes=$8    # number of evaluation episodes, default 100


# #
# #
# #
# python main.py \
#     --seed $seed \
#     --data_name "toy" \
#     --env_name "toy" \
#     --ret_file "offline_data/${algo}_toy_seed${seed}" \
#     --device $device \
#     --algo $algo \
#     --config "configs/${algo}/test.yaml" \
#     --checkpoint_dir "checkpoints/${algo}_toy_seed${seed}" \
#     --model_type "dt" \
#     --K 4 \
#     --train_iters 2 \
#     --num_steps_per_iter 10


# #
# #
# #
# python main.py \
#     --seed $seed \
#     --data_name "mstoy" \
#     --env_name "mstoy" \
#     --ret_file "offline_data/${algo}_mstoy_seed${seed}" \
#     --device $device \
#     --algo $algo \
#     --config "configs/${algo}/test.yaml" \
#     --checkpoint_dir "checkpoints/${algo}_mstoy_seed${seed}" \
#     --model_type "dt" \
#     --K 4 \
#     --train_iters 2 \
#     --num_steps_per_iter 10


# #
# #
# #
# for learner_opt in 70
# do
#     for adv_opt in 50
#     do
#         d_name="c4data_mdp_${learner_opt}_mdp_${adv_opt}"
#         ret_file="offline_data/${algo}${mode}_${d_name}_new_new_seed${seed}_al${alpha}_${seed}"

#         # Adjust the test-time adversary by changing --test_adv
#         python main.py \
#             --seed $seed \
#             --data_name $d_name \
#             --env_name "connect_four" \
#             --ret_file $ret_file \
#             --device $device \
#             --algo $algo \
#             --config "configs/${algo}/test.yaml" \
#             --checkpoint_dir "checkpoints/${algo}_${model_type}_${d_name}_${alpha}_seed${seed}" \
#             --model_type "dt" \
#             --K 22 \
#             --train_iters 1 \
#             --num_steps_per_iter 100 \
#             --test_adv 0.5 \
#             --num_eval_episodes 3
#     done
# done


# #
# #
# #
# d_name="arrl_train_${env_name}_high"
# added_data_name="random_${env_name}"
# adv_base="arrl/models/"
# adv="${adv_base}${env_name}/ou_noise/nr_mdp_0.1_1/0"

# # mix_coef here defines the proportion between collected and random data
# for added_data_prop in "${mix_coef}"
# do
#     ret_file="offline_data/${algo}_${d_name}_${added_data_name}_${added_data_prop}"

#     # Training
#     python main.py \
#         --is_training_only \
#         --seed $seed \
#         --data_name $d_name \
#         --added_data_name $added_data_name \
#         --added_data_prop $added_data_prop \
#         --env_name $env_name \
#         --ret_file $ret_file \
#         --device $device \
#         --algo $algo \
#         --config "configs/${algo}/test.yaml" \
#         --checkpoint_dir "checkpoints/${algo}_${model_type}_${d_name}_${added_data_name}_${added_data_prop}_seed${seed}" \
#         --K 20 \
#         --model_type $model_type \
#         --train_iters 2 \
#         --num_steps_per_iter 5 \
#         --test_adv $adv \
#         --env_alpha $env_alpha \
#         --num_eval_episodes 0

#     # Testing
#     python main.py \
#         --is_testing_only \
#         --seed $seed \
#         --data_name $d_name \
#         --added_data_name $added_data_name \
#         --added_data_prop $added_data_prop \
#         --env_name $env_name \
#         --ret_file $ret_file \
#         --device $device \
#         --algo $algo \
#         --config "configs/${algo}/test.yaml" \
#         --checkpoint_dir "checkpoints/${algo}_${model_type}_${d_name}_${added_data_name}_${added_data_prop}_seed$(( $seed % 5 ))" \
#         --K 20 \
#         --model_type $model_type \
#         --train_iters 0 \
#         --num_steps_per_iter 0 \
#         --test_adv $adv \
#         --env_alpha $env_alpha \
#         --num_eval_episodes 3
# done
