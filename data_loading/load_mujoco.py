import copy
import pickle

import gymnasium as gym
import numpy as np
from datasets import Dataset

from arrl.main import load_env_name
from data_loading.dataclasses import AdvGymEnv, Trajectory


def _get_hf_offline_data(dir_prefix: str, traj_length: int | None = None) -> list[Trajectory]:
    """
    Load the offline HuggingFace dataset from the given directory.

    Args:
        dir_prefix (str): The directory prefix of the dataset.
        traj_length (int | None): The length of the trajectory to use. If None, use the full trajectory.

    Returns:
        trajs (List[Trajectory]): The list of trajectories.
    """
    raw_trajs = Dataset.from_file(f"{dir_prefix}/data-00000-of-00001.arrow", split='train')
    trajs = []
    for raw_traj in raw_trajs:
        raw_traj['observations'] = raw_traj.pop('state')
        raw_traj['pr_actions'] = raw_traj.pop('pr_action')
        raw_traj['dones'] = raw_traj.pop('done')
        raw_traj['rewards'] = raw_traj.pop('reward')
        if 'adv_actions' not in raw_traj:
            raw_traj['adv_actions'] = raw_traj.pop('adv_action')
        infos_ = [{'adv': adv_act} for adv_act in raw_traj['adv_actions']]
        length = len(infos_) if not traj_length else traj_length
        trajs.append(Trajectory(
            obs=raw_traj['observations'][:length], 
            actions=raw_traj['pr_actions'][:length], 
            rewards=np.array(raw_traj['rewards'][:length]),
            infos=infos_[:length], 
            policy_infos=[]
        ))
    return trajs


def load_mujoco_env(
        env_name: str, 
        dir_prefix: str = '', 
        adv_model_path: str = '', 
        added_dir_prefix: str | None = None,
        added_data_prop: float = 1.0, 
        env_alpha: float = 0.1,
        traj_length: int = 1000, 
        device_str: str = "cpu", 
    ) -> tuple[AdvGymEnv, list[Trajectory]]:
    """
    Load the Mujoco environment, and corresponding HuggingFace offline dataset.

    Args:
        env_name (str): The name of the environment.
        dir_prefix (str): The directory prefix of the dataset.
        adv_model_path (str): The path to the adversarial model.
        added_dir_prefix (str): The directory prefix of the added dataset.
        added_data_prop (float): The proportion of the added dataset.
        env_alpha (float): The alpha value of the environment.
        traj_length (int): The length of the trajectory to use.
        device (str): The device to use.

    Returns:
        env (AdvGymEnv): The environment.
        trajs (List[Trajectory]): The list of trajectories.
    """
    # Load the basic environment and body mass configuration
    basic_env = gym.make(load_env_name(env_name))
    basic_bm = copy.deepcopy(basic_env.env.env.model.body_mass.copy())
    device = torch.device(device_str)
    env = AdvGymEnv(basic_env, adv_model_path, device, env_name, env_alpha, basic_bm)
    trajs = _get_hf_offline_data(dir_prefix, traj_length)

    # Load the added data, if any
    added_trajs = []
    if added_dir_prefix:
        with open(added_dir_prefix, 'rb') as file:
            added_trajs = pickle.load(file)

    # Combine the datasets and return
    if added_data_prop == -1:
        return env, trajs + added_trajs
    elif added_data_prop > 1:
        return env, trajs[:int(len(added_trajs) / added_data_prop)] + added_trajs
    else:
        return env, trajs + added_trajs[:int(len(trajs) * added_data_prop)]
