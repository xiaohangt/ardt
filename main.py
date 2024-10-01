import argparse
import random
import sys

import gym
import numpy as np
import torch

from data_loading.load_mujoco import load_mujoco_env, Trajectory
from decision_transformer.experiment import experiment
from return_transforms.generate import generate_expected, generate_maxmin

from stochastic_offline_envs.envs.offline_envs.connect_four_offline_env import ConnectFourOfflineEnv
from stochastic_offline_envs.envs.connect_four.connect_four_env import GridWrapper
from stochastic_offline_envs.envs.offline_envs.gambling_offline_env import GamblingOfflineEnv
from stochastic_offline_envs.envs.offline_envs.toy_offline_env import ToyOfflineEnv
from stochastic_offline_envs.envs.offline_envs.mstoy_offline_env import MSToyOfflineEnv
from stochastic_offline_envs.envs.offline_envs.mstoy_offline_env import MSToyOfflineEnv


MUJOCO_TARGETS_DICT = {'halfcheetah': [2000, 3000], 'hopper': [500, 1000], 'walker2d': [800, 1000]}


def set_seed_everywhere(seed: int, env: int | None = None):
    """
    Set seed for every possible source of randomness
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.mps.manual_seed(seed)
    if env is not None:
        env.seed = seed
        env.action_space.seed = seed


def process_c4_trajs(
        task: ConnectFourOfflineEnv, 
        traj_len: int, 
        added_data_name: str
    ) -> list[Trajectory]:
    """
    Process Connect Four trajectories to conform to desired format.
    """
    # Limit episode length
    task.trajs = task.trajs[:traj_len]
    trajs = task.trajs 
    # Add further similar-sized trajectories
    if added_data_name:
        task_added = ConnectFourOfflineEnv(
            data_name=added_data_name, test_regen_prob=eval(test_adv)
        )
        trajs += task_added.trajs[:traj_len]
    # Force it to conform to desired format
    for traj in trajs:
        for i in range(len(traj.obs)):
            traj.obs[i] = traj.obs[i]['grid']
    return trajs


def load_env(
        env_name: str, 
        traj_len: int | None = None, 
        data_name: str = "", 
        added_data_name: str = "",
        added_data_prop: float = 1.0,
        test_adv: str = '0.0', 
        env_alpha: float = 0.1
    ) -> tuple[gym.Env, list[Trajectory], dict]:
    """
    Load environment and trajectories.

    Args:
        env_name: Name of the environment.
        traj_len: Number of trajectories to load.
        data_name: Name of the dataset.
        added_data_name: Name of the dataset to add.
        added_data_prop: Proportion of added data.
        test_adv: Adversarial parameter.
        env_alpha: Environmental parameter.
    
    Returns:
        env: Gym environment.
        trajs: List of trajectories.
        env_params: Dictionary of environment parameters.
    """
    print(f'Loading task: {env_name}')

    if 'connect_four' in env_name:
        task = ConnectFourOfflineEnv(
            data_name=data_name, test_regen_prob=eval(test_adv)
        )
        env = GridWrapper(task.env_cls())
        trajs = process_c4_trajs(
            task=task,
            traj_len=(traj_len if traj_len is not None else len(task.trajs)),
            added_data_name=added_data_name
        )
        env_params = {
            "task": task, 
            "max_ep_len": 22, 
            "env_targets": [1, 1.5, 2], 
            "scale": 10, 
            "action_type": "discrete"
        }
    elif env_name == 'gambling':
        task = GamblingOfflineEnv()
        env = task.env_cls()
        trajs = task.trajs
        env_params = {
            "task": task, 
            "max_ep_len": 5, 
            "env_targets": list(np.arange(-15, 5.01, 0.5)), 
            "scale": 5, 
            "action_type": "discrete"
        }
    elif env_name == 'toy':
        task = ToyOfflineEnv()
        env = task.env_cls()
        trajs = task.trajs
        env_params = {
            "task": task, 
            "max_ep_len": 5, 
            "env_targets": list(np.arange(0, 6.01, 0.5)), 
            "scale": 5, 
            "action_type": "discrete"
        }
    elif "mstoy" in env_name:
        task = MSToyOfflineEnv() 
        env = task.env_cls()
        trajs = task.trajs
        env_params = {
            "task": task, 
            "max_ep_len": 5, 
            "env_targets": list(np.arange(0, 7.01, 0.5)), 
            "scale": 5, 
            "action_type": "discrete"
        }
    elif env_name in ['halfcheetah', 'hopper', 'walker2d']:
        task, env, trajs = load_mujoco_env(
            env_name, 
            dir_prefix=f"stochastic_offline_envs/offline_data/{data_name}", 
            adv_model_path=test_adv, 
            added_dir_prefix= f"stochastic_offline_envs/offline_data/{added_data_name}",
            added_data_prop=added_data_prop,
            env_alpha=env_alpha
        )
        env_params = {
            "task": task, 
            "max_ep_len": 1000, 
            "env_targets": MUJOCO_TARGETS_DICT[env_name], 
            "scale": 1000, 
            "action_type": "continuous"
        }
    
    print(f"Finished loading task with {len(trajs)} trajectories.")
    return (
        env,
        trajs,
        env_params
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_name', type=str, required=True)
    parser.add_argument('--added_data_name', type=str, default='')
    parser.add_argument('--added_data_prop', type=float, default=0.0)
    parser.add_argument('--env_name', type=str, required=True, choices=['toy', 'mstoy', 'connect_four', 'halfcheetah', 'hopper', 'walker2d'])
    parser.add_argument('--ret_file', type=str, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--n_cpu', type=int, default=1)

    # For returns transformation: 
    parser.add_argument('--algo', type=str, required=True, choices=['ardt', 'dt', 'esper', 'bc'])
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--is_simple_maxmin_model', action='store_true')

    # For decision transformer training:
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--is_collect_data_only', action='store_true')
    parser.add_argument('--is_relabeling_only', action='store_true')
    parser.add_argument('--is_training_only', action='store_true')
    parser.add_argument('--is_testing_only', action='store_true')

    parser.add_argument('--traj_len', type=int, default=None)
    parser.add_argument('--top_pct_traj', type=float, default=1.)

    parser.add_argument('--model_type', type=str, default='dt', choices=['adt', 'dt', 'bc'])
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--grad_clip_norm', type=float, default=0.25)

    parser.add_argument('--train_iters', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--argmax', type=bool, default=False)
    parser.add_argument('--rtg_seq', type=bool, default=True)
    parser.add_argument('--normalize_states', action='store_true')
    
    # For decision transformer evaluation:
    parser.add_argument('--env_data_dir', type=str, default="")
    parser.add_argument('--test_adv', type=str, default='0.8')
    parser.add_argument('--env_alpha', type=float, default=0.1)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    
    # Process args and check for consistency
    args = parser.parse_args()
    variant = vars(args)
    if variant['algo'] == 'bc':
        assert variant['model_type'] == 'bc', "Behavioural Cloning algo requires BC model type"
    if variant['device'] == 'gpu':
        variant['device'] = 'cuda'
    set_seed_everywhere(variant['seed'])
    print(f"Running with arguments:\n{variant}")

    print("############### Loading Environment ##################")
    env, offline_trajs, env_params = load_env(
        env_name=variant['env_name'], 
        traj_len=variant['traj_len'],
        data_name=variant['data_name'], 
        added_data_name=variant['added_data_name'],
        added_data_prop=variant['added_data_prop'],
        test_adv=variant['test_adv'],
        env_alpha=variant['env_alpha']
    )
    
    if variant['is_collect_data_only']:
        # if the flag is set to only collect data, exit after loading the environment
        sys.exit(0)

    if not variant['is_training_only']:
        # if not only training the protagonist, (re-)do the relabeling process
        print("############### Relabeling Returns Data ###############")
        print(f"Will save relabeled file to {variant['ret_file']}")
    
        if variant['algo'] == 'ardt':
            generate_maxmin(
                env, 
                offline_trajs, 
                variant['config'], 
                variant['ret_file'], 
                variant['device'], 
                variant['n_cpu'], 
                is_simple_model=variant['is_simple_maxmin_model'],
                is_toy=(variant['env_name'] == 'toy')
            )
        elif variant['algo'] == 'esper':
            generate_expected(
                env, 
                offline_trajs, 
                variant['config'], 
                variant['ret_file'], 
                variant['device'], 
                variant['n_cpu']
            )
        elif variant['algo'] == 'dt' or variant['algo'] == 'bc':
            pass
        else:
            raise NotImplementedError('Chosen relabeling algorithm unknown.')

    if not variant['is_relabeling_only']:
        # if not only relabeling trajectory returns, start the protagonist training process
        print("############### Training Decision Transformer ###############")

        test_advs = []
        if not variant['is_training_only']:
            test_advs = [variant['test_adv']]
            if variant['env_name'] in ['halfcheetah', 'hopper', 'walker2d']:
                if "env" not in variant['test_adv']:
                    test_advs = [variant['test_adv'][:-1] + str(adv) for adv in range(8)]
                else:
                    test_advs = [f"env{mass}" for mass in [0.5, 0.7, 1.0, 1.5, 2.0]]

        for test_adv in test_advs:   
            variant['test_adv'] = test_adv
            if variant['env_name'] in {'halfcheetah', 'hopper', 'walker2d'}:
                env.reset_adv_agent(variant['test_adv'], variant['device'])

            experiment(
                env_params['task'],
                env,
                env_params['max_ep_len'],
                env_params['env_targets'],
                env_params['scale'],
                env_params['action_type'],
                variant=vars(args)
            )
