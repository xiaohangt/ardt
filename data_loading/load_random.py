import copy
import pickle

import gymnasium as gym
import numpy as np
import torch

from utils import AdvGymEnv
from utils import Trajectory

from arrl.ddpg import DDPG
from arrl.main import load_env_name
from arrl.utils import load_model 


def collect_random_data(env_name, device, adv_model_path, data_path, horizon=1000):
    actual_adv_path = adv_model_path.replace(env_name, load_env_name(env_name))
    env = gym.make(load_env_name(env_name))
    agent = DDPG(
        gamma=1, 
        tau=1, 
        hidden_size=64, 
        num_inputs=env.observation_space.shape[0],
        action_space=env.action_space.shape[0], 
        train_mode=False, 
        alpha=1, 
        replay_size=1, 
        device=device
    )
        
    load_model(agent, basedir=actual_adv_path)
    agent.eval()
    noise = torch.distributions.uniform.Uniform(torch.tensor([-1.0], dtype=torch.float32), torch.tensor([1.0], dtype=torch.float32))

    n_interactions = 1e6
    trajs = []
    n_gathered = 0

    obs_ = []
    actions_ = []
    rewards_ = []
    infos_ = []
    policy_infos_ = []
    t = 0

    obs = env.reset()
    reward = None

    with torch.no_grad():
        while n_gathered < n_interactions:
            if n_gathered % 1e4 == 0:
                print(f"\r{n_gathered} steps done", end='')
            obs_.append(copy.deepcopy(obs))

            state = torch.from_numpy(obs).to(device, dtype=torch.float32)

            pr_action = agent.actor(state).clamp(-1, 1).cpu()
            if np.random.random() < 0.1:
                pr_action = noise.sample(pr_action.shape).view(pr_action.shape).cpu()

            adv_action = agent.adversary(state).clamp(-1, 1).cpu()
            if np.random.random() < 0.1:
                adv_action = noise.sample(adv_action.shape).view(adv_action.shape).cpu()

            step_action = (pr_action * 0.9 + adv_action * 0.1).data.clamp(-1, 1).numpy()

            policy_infos_.append({})
            actions_.append(pr_action.numpy())

            obs, reward, done, _ = env.step(step_action)

            t += 1
            infos_.append({"adv": adv_action.numpy()})
            rewards_.append(reward)

            n_gathered += 1

            if t == horizon or done:
                trajs.append(Trajectory(
                    obs=obs_,
                    actions=actions_,
                    rewards=rewards_,
                    infos=infos_,
                    policy_infos=policy_infos_)
                )
                t = 0
                obs_ = []
                actions_ = []
                rewards_ = []
                infos_ = []
                policy_infos_ = []
                obs = env.reset()
                reward = None    

    with open(data_path, 'wb') as file:
        pickle.dump(trajs, file)
        print('Saved trajectories to dataset file', data_path)  
    
    return AdvGymEnv(gym.make(load_env_name(env_name)), actual_adv_path, device), trajs
