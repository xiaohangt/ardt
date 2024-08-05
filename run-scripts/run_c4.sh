#!/bin/bash
seed=$1
algo=$2       # ardt, dt, esper

env_name="connect_four"
device=0

model_type=dt
ll=0.9

# *_opt here is optimality percentage, equal to (1 - epsilon) * 100%
for learner_opt in 70 60 50
do
    for adv_opt in 50 70 90
    do
        d_name="c4data_mdp_${learner_opt}_mdp_${adv_opt}"
        ret_file="../data/${algo}${mode}_${d_name}_new_new_ll${ll}_al${alpha}_${seed}"

        # Adjust the test-time adversary by changing --test_adv
        python main.py \
        --test_adv 0.5 \
        --model_type $model_type \
        --device "cuda:${device}" \
        --alpha $alpha \
        --data_name $d_name \
        --env_name $env_name \
        --algo $algo \
        --ret_file $ret_file \
        --seed $seed \
        --leaf_weight $ll \
        --config "../configs/${algo}/connect_four.yaml" \
        --max_iters 1 \
        --num_steps_per_iter 100000 \
        --num_eval_episodes 100 \
        --K 22 \
        --checkpoint_dir "../checkpoints/${algo}_${model_type}_${d_name}_${alpha}_seed${seed}"

    done
done


done

