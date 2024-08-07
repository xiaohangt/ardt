import pickle
import yaml
from collections import defaultdict
from pathlib import Path

import numpy as np

from return_transforms.algos.esper.esper import esper
from return_transforms.algos.maxmin.maxmin import worst_case_qf


def load_config(config_path):
    return yaml.safe_load(Path(config_path).read_text())


def normalize_obs(trajs):
    obs_list = []
    for traj in trajs:
        obs_list.extend(traj.obs)
    obs = np.array(obs_list)
    obs_mean = np.mean(obs, axis=0)
    obs_std = np.std(obs, axis=0) + 1e-8
    for traj in trajs:
        for i in range(len(traj.obs)):
            traj.obs[i] = (traj.obs[i] - obs_mean) / obs_std
    return trajs


def generate(env, trajs, config, ret_file, device, n_cpu=2, ):
    print('Loading config...')
    config = load_config(config)

    if config['method'] == 'esper':
        print('Loading offline RL task...')

        if config['normalize']:
            print('Normalizing observations...')
            trajs = normalize_obs(trajs)

        print('Creating ESPER returns...')
        rets = esper(
            trajs,
            env.action_space,
            config['dynamics_model_args'],
            config['cluster_model_args'],
            config['train_args'],
            device,
            n_cpu
        )

        # Save the returns as a pickle
        print('Saving returns...')
        Path(ret_file).parent.mkdir(parents=True, exist_ok=True)
        with open(ret_file, 'wb') as f:
            pickle.dump(rets, f)
    else:
        raise NotImplementedError


def generate_maxmin(env, env_name, trajs, config, ret_file, device, n_cpu, lr, wd, is_simple_model=False, batch_size=64, leaf_weight=0.5, alpha=0.01):
    print('Loading config...')
    config = load_config(config)

    if config['method'] == 'ardt':
        if config['normalize']:
            print('Normalizing observations...')
            trajs = normalize_obs(trajs)

        print('Creating ARDT returns...')
        rets, prompt_value, qsa2_model = worst_case_qf(
            env_name,
            trajs,
            env.action_space,
            env.adv_action_space,
            config['train_args'],
            device,
            n_cpu,
            lr,
            wd,
            is_simple_model,
            batch_size,
            leaf_weight=leaf_weight,
            alpha=alpha
        )

        # Save the returns as a pickle
        print(f'Saving returns to {ret_file}')
        Path(ret_file).parent.mkdir(parents=True, exist_ok=True)
        with open(f"{ret_file}", 'wb') as f:
            pickle.dump(rets, f)
        
        print(f'Saving prompt to {f"{ret_file}_prompt.pkl"}')
        with open(f"{ret_file}_prompt.pkl", 'wb') as f:
            pickle.dump(prompt_value, f)
    else:
        raise NotImplementedError
    

def get_stats(rets, trajs):
    results = defaultdict(list)
    final_res = {}
    for i in range(len(trajs)):
        results[trajs[i].actions[0]].append(rets[i][0])
    for key in results.keys():
        final_res[key] = np.mean(results[key])
    return final_res
