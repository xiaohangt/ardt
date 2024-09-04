import pickle
from pathlib import Path

import numpy as np
import yaml

import gym

from data_loading.load_mujoco import Trajectory
from return_transforms.algos.esper.esper import esper
from return_transforms.algos.maxmin.maxmin import maxmin


def _normalize_obs(trajs: list[Trajectory]) -> list[Trajectory]:
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


def generate_expected(
        env: gym.Env, 
        trajs: list[Trajectory], 
        config: dict, 
        ret_file: str, 
        device: str, 
        n_cpu: int
    ):
    config = yaml.safe_load(Path(config).read_text())
    assert config['method'] == 'esper', "ESPER is the algo to use to learn expected returns."

    if config['normalize']:
        trajs = _normalize_obs(trajs)

    print('Generating ESPER returns...')
    rets = esper(
        trajs,
        env.action_space,
        config['dynamics_model_args'],
        config['cluster_model_args'],
        config['train_args'],
        device,
        n_cpu
    )

    print(f'Done. Saving returns to {ret_file}.')
    Path(ret_file).parent.mkdir(parents=True, exist_ok=True)
    with open(f"{ret_file}.pkl", 'wb') as f:
        pickle.dump(rets, f)


def generate_maxmin(
        env: gym.Env, 
        trajs: list[Trajectory], 
        config: dict, 
        ret_file: str, 
        device: str, 
        n_cpu: int,
        is_simple_model: bool = False, 
        is_toy: bool = False
    ):
    config = yaml.safe_load(Path(config).read_text())
    assert config['method'] == 'ardt', "ARDT is the algo to use to learn worst-case returns."

    if config['normalize']:
        trajs = _normalize_obs(trajs)

    print('Generating ARDT returns...')
    rets, prompt_value = maxmin(
        trajs,
        env.action_space,
        env.adv_action_space,
        config['train_args'],
        device,
        n_cpu,
        is_simple_model=is_simple_model,
        is_toy=is_toy
    )

    print(f'Done. Saving returns and prompts to {ret_file}.')
    Path(ret_file).parent.mkdir(parents=True, exist_ok=True)
    with open(f"{ret_file}.pkl", 'wb') as f:
        pickle.dump(rets, f)
    with open(f"{ret_file}_prompt.pkl", 'wb') as f:
        pickle.dump(prompt_value, f)
    