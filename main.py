import argparse
import random
from collections import namedtuple

import numpy as np
import torch

from decision_transformer.experiment import experiment
from data_loading.load_mujoco import load_mujoco_env
from return_transforms.generate import generate, generate_maxmin
from stochastic_offline_envs.envs.offline_envs.connect_four_offline_env import ConnectFourOfflineEnv
from stochastic_offline_envs.envs.connect_four.connect_four_env import GridWrapper
from stochastic_offline_envs.envs.offline_envs.gambling_offline_env import GamblingOfflineEnv
from stochastic_offline_envs.envs.offline_envs.toy_offline_env import ToyOfflineEnv
from stochastic_offline_envs.envs.offline_envs.mstoy_offline_env import MSToyOfflineEnv
from stochastic_offline_envs.envs.offline_envs.mstoy_offline_env import MSToyOfflineEnv


def set_seed_everywhere(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.mps.manual_seed(seed)
    if env is not None:
        env.seed = seed
        env.action_space.seed = seed


def load_env(env_name, 
             traj_len, 
             data_name=None, 
             data_dir="offline_data",
             test_adv='0.0', 
             added_data_name="",
             added_data_prop=1.0,
             env_alpha=0.1):
    
    print(f'Loading offline RL task: {env_name}')
    max_ep_len, env_targets, scale, action_type = 1000, [1, 1.5], 1., "discrete"
    if 'connect_four' in env_name:
        max_ep_len, scale = 22, 10
        task = ConnectFourOfflineEnv(data_name=data_name, 
                                    test_regen_prob=eval(test_adv))
        env = task.env_cls()
        env = GridWrapper(env)
        if traj_len:
            task.trajs = task.trajs[:traj_len]
        trajs = task.trajs 
        if added_data_name:
            task_added = ConnectFourOfflineEnv(data_name=added_data_name, 
                                              test_regen_prob=eval(test_adv))
            trajs += task_added.trajs
        for traj in trajs:
            for i in range(len(traj.obs)):
                traj.obs[i] = traj.obs[i]['grid']
        data_name = data_name + added_data_name
    elif env_name == 'gambling':
        task = GamblingOfflineEnv()
        max_ep_len, env_targets, scale = 5, list(np.arange(-15, 5, 0.5)) + [5.], 5.
        env = task.env_cls()
        trajs = task.trajs
    elif env_name == 'toy':
        task = ToyOfflineEnv()
        max_ep_len, env_targets, scale = 5, list(np.arange(0, 6, 0.5)) + [6., 10.], 5.
        env = task.env_cls()
        trajs = task.trajs
    elif "mstoy" in env_name :
        task = MSToyOfflineEnv() 
        max_ep_len, env_targets, scale = 5, list(np.arange(0, 7, 0.5)) + [7., 10.], 5. 
        env = task.env_cls()
        trajs = task.trajs
    elif env_name in ['halfcheetah', 'hopper', 'walker2d']:
        scale = 1000.
        tr_dicts = {'halfcheetah': [2000, 3000], 'hopper': [500, 1000], 'walker2d': [800, 1000]}
        dir_prefix = f"offline_data/{data_name}" 
        added_dir_prefix = f"offline_data/{added_data_name}" 

        env, trajs = load_mujoco_env(env_name, 
                                     data_name, 
                                     dir_prefix=dir_prefix, 
                                     adv_model_path=test_adv, 
                                     added_dir_prefix=added_dir_prefix,
                                     added_data_prop=added_data_prop,
                                     env_alpha=env_alpha
                                     )
        max_ep_len = 1000
        env_targets = tr_dicts[env_name]
        task = namedtuple("Task", ["trajs", "test_env_cls"])
        task.trajs = trajs
        task.test_env_cls = lambda: env
        action_type = "continuous"
        data_name = data_name + added_data_name
        
    print(f'Finished loading offline RL task: {env_name}', len(trajs))

    return task, max_ep_len, env_targets, scale, action_type, env, trajs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_name', type=str, required=True)
    parser.add_argument('--added_data_name', type=str, default='')
    parser.add_argument('--added_data_prop', type=float, default=0.0)
    parser.add_argument('--env_name', type=str, required=True, choices=['toy', 'mstoy', 'connect_four', 'halfcheetah', 'hopper', 'walker2d'])
    parser.add_argument('--ret_file', type=str, required=True)
    parser.add_argument('--device', type=str, required=True)

    # for return transformation: 
    parser.add_argument('--algo', type=str, required=True, choices=['ardt', 'dt', 'esper', 'bc'])
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--leaf_weight', type=float, default=0.9)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=1e-3) 
    parser.add_argument('--wd', type=float, default=1e-4) 
    parser.add_argument('--n_cpu', type=int, default=1)
    parser.add_argument('--is_simple_model', action='store_true')

    # for decision transformer:
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--is_training', action='store_true')
    parser.add_argument('--is_testing', action='store_true')
    parser.add_argument('--is_relabeling', action='store_true')
    parser.add_argument('--collect_data', action='store_true')
    parser.add_argument('--prop_data', type=float, default=1.)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--traj_len', type=int, default=None)

    parser.add_argument('--model_type', type=str, default='dt', choices=['adt', 'dt', 'bc'])
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--argmax', type=bool, default=False)
    parser.add_argument('--rtg_seq', type=bool, default=True)
    parser.add_argument('--normalize_states', action='store_true')
    
    # for decision transformer evaluation:
    parser.add_argument('--env_data_dir', type=str, default="")
    parser.add_argument('--test_adv', type=str, default='0.8')
    parser.add_argument('--env_alpha', type=float, default=0.1)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    

    args = parser.parse_args()
    variant = vars(args)
    if variant['algo'] == 'bc':
        assert variant['model_type'] == 'bc'
    if variant['device'] == 'gpu':
        variant['device'] = 'cuda'
    
    print("Arguments:", variant)

    print("############### Loading ##################")
    task, max_ep_len, env_targets, scale, action_type, env, trajs = \
        load_env(env_name=args.env_name, 
            traj_len=args.traj_len,
            data_name=args.data_name, 
            data_dir=args.env_data_dir, 
            test_adv=args.test_adv,
            added_data_name=args.added_data_name,
            added_data_prop=args.added_data_prop,
            env_alpha=args.env_alpha)

    if not args.collect_data:
        print("############### Relabeling ###############")
        print("Will save relabeled file to", args.ret_file)
        set_seed_everywhere(args.seed)
        
        if not args.is_training:
            config, ret_file, device, n_cpu, lr, wd = args.config, args.ret_file, args.device, args.n_cpu, args.lr, args.wd
            if args.algo == 'ardt':
                generate_maxmin(env, args.env_name, trajs, config, ret_file, device, n_cpu, lr, wd, args.is_simple_model, args.batch_size, args.leaf_weight, args.alpha)
            elif args.algo == 'dt' or args.algo == 'bc':
                pass
            elif args.algo == 'esper':
                generate(env, trajs, config, ret_file, device, n_cpu=2)
            else:
                raise Exception('Algo error')

        print("############### Training ###############")
        print()

        advs = [args.test_adv]
        testing_only = variant['max_iters'] == 0 or variant['num_steps_per_iter'] == 0
        if variant['is_training'] and testing_only and args.env_name in ['halfcheetah', 'hopper', 'walker2d']:
            if "env" not in args.test_adv:
                advs = [args.test_adv[:-1] + str(adv) for adv in range(8)]
            else:
                advs = [f"env{mass}" for mass in [0.5, 0.7, 1.0, 1.5, 2.0]]

        for test_adv in advs:   
            args.test_adv = test_adv
            if args.env_name in ['halfcheetah', 'hopper', 'walker2d']:
                env.reset_model_rl(test_adv, args.device)
            if not args.is_relabeling:
                experiment(task,
                            env,
                            max_ep_len,
                            env_targets,
                            scale,
                            action_type,
                            variant=vars(args))