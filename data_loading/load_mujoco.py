import copy
import pickle
from collections import namedtuple

import gym
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
        if 'state' in raw_traj:
            raw_traj['observations'] = raw_traj.pop('state')
        if 'pr_action' in raw_traj:
            raw_traj['pr_actions'] = raw_traj.pop('pr_action')
        if 'done' in raw_traj:
            raw_traj['dones'] = raw_traj.pop('done')
        if 'reward' in raw_traj:
            raw_traj['rewards'] = raw_traj.pop('reward')
        if 'adv_action' in raw_traj:
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
        added_data_prop: float = 0.0, 
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
        task (Task): The task at hand.
        env (AdvGymEnv): The environment.
        trajs (List[Trajectory]): The list of trajectories.
    """
    # Load the basic environment and body mass configuration
    basic_env = gym.make(load_env_name(env_name))
    basic_bm = copy.deepcopy(basic_env.env.env.model.body_mass.copy())
    env = AdvGymEnv(basic_env, env_name, adv_model_path, basic_bm, env_alpha, device_str)
    trajs = _get_hf_offline_data(dir_prefix, traj_length)

    # Load the added data, if any
    added_trajs = []
    if added_dir_prefix and added_data_prop != 0:
        with open(added_dir_prefix, 'rb') as file:
            added_trajs = pickle.load(file)

    # Build the task itself with required trajectories
    task = namedtuple("Task", ["trajs", "test_env_cls"])
    task.test_env_cls = lambda: env
    if added_data_prop == -1:
        task.trajs = trajs + added_trajs
    elif added_data_prop > 1:
        task.trajs = trajs[:int(len(added_trajs) / added_data_prop)] + added_trajs
    else:
        task.trajs = trajs + added_trajs[:int(len(trajs) * added_data_prop)]
    
    return task, env, task.trajs
