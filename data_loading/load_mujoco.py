import copy
import pickle

import gymnasium as gym
import numpy as np
from datasets import Dataset

from arrl.ddpg import DDPG
from arrl.main import load_env_name
from data_loading.utils import AdvGymEnv, Trajectory


def get_hf_offline_data(dir_prefix, used_length=None):
    raw_trajs = Dataset.from_file(dir_prefix + "/data-00000-of-00001.arrow", split='train')
    trajs = []
    print("HF Offline Dataset loading:", dir_prefix)
    dones_list = []
    for raw_traj in raw_trajs:
        if 'adv_actions' not in raw_traj:
            raw_traj['observations'] = raw_traj.pop('state')
            raw_traj['pr_actions'] = raw_traj.pop('pr_action')
            raw_traj['adv_actions'] = raw_traj.pop('adv_action')
            raw_traj['dones'] = raw_traj.pop('done')
            raw_traj['rewards'] = raw_traj.pop('reward')
        infos_ = [{'adv': adv_act} for adv_act in raw_traj['adv_actions']]
        length = len(infos_) if not used_length else used_length
        dones_list.append(np.sum(raw_traj['dones']))
        trajs.append(Trajectory(
            obs=raw_traj['observations'][:length], 
            actions=raw_traj['pr_actions'][:length], 
            rewards=np.array(raw_traj['rewards'][:length]),
            infos=infos_[:length], 
            policy_infos=[]
        ))
    return trajs


def load_mujoco_env(
        env_name, 
        data_name=None, 
        used_length=1000, 
        device="cpu", 
        adv_model_path='', 
        dir_prefix='', 
        added_dir_prefix=None,
        added_data_prop=1.0, 
        env_alpha=0.1
    ):
    basic_env = gym.make(load_env_name(env_name))
    basic_bm = copy.deepcopy(basic_env.env.env.model.body_mass.copy())
    env = AdvGymEnv(basic_env, adv_model_path, device, env_name, env_alpha, basic_bm)
    trajs = get_hf_offline_data(dir_prefix, used_length)

    added_trajs = []
    if added_dir_prefix:
        with open(added_dir_prefix, 'rb') as file:
            added_trajs = pickle.load(file)

    if added_data_prop == -1:
        return env, trajs + added_trajs
    elif added_data_prop > 1:
        return env, trajs[:int(len(added_trajs) / added_data_prop)] + added_trajs
    else:
        return env, trajs + added_trajs[:int(len(trajs) * added_data_prop)]
