#!/bin/bash
seed=$1   # desired seed
algo=$2   # ardt, dt, esper
device=$3 # cpu, cuda

# *_opt here is optimality percentage, equal to (1 - epsilon) * 100%
for learner_opt in 70 60 50
do
    for adv_opt in 50 70 90
    do
        d_name="c4data_mdp_${learner_opt}_mdp_${adv_opt}"
        ret_file="offline_data/${algo}${mode}_${d_name}_new_new_seed${seed}_al${alpha}_${seed}"

        # Adjust the test-time adversary by changing --test_adv
        python main.py \
            --seed $seed \
            --data_name $d_name \
            --env_name "connect_four" \
            --ret_file $ret_file \
            --device $device \
            --algo $algo \
            --config "configs/${algo}/connect_four.yaml" \
            --checkpoint_dir "checkpoints/${algo}_${model_type}_${d_name}_${alpha}_seed${seed}" \
            --model_type "dt" \
            --K 22 \
            --train_iters 1 \
            --num_steps_per_iter 100000 \
            --test_adv 0.5 \
            --num_eval_episodes 100
    done
done
