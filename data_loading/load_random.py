import copy
import pickle

import gym
import numpy as np
import torch

from arrl.ddpg import DDPG
from arrl.main import load_env_name
from arrl.utils import load_model 
from data_loading.dataclasses import AdvGymEnv
from data_loading.dataclasses import Trajectory


def collect_random_data(
        env_name: str, 
        data_path: str, 
        adv_model_path: str, 
        traj_length: int = 1000,
        n_transitions: int = 1e6,
        rnd_prob: float = 0.1,
        env_alpha: float = 0.1,
        device_str: str = "cpu", 
    ) -> tuple[AdvGymEnv, list[Trajectory]]:
    """
    Collect random data from the given environment and adversarial model.

    Args:
        env_name (str): The name of the environment.
        data_path (str): The path to save the data.
        adv_model_path (str): The path to the adversarial model.
        traj_length (int): The length of the trajectory.
        device (str): The device to use.

    Returns:
        env (AdvGymEnv): The environment.
        trajs (List[Trajectory]): The list of trajectories.
    """
    # Load the environment
    env = gym.make(load_env_name(env_name))

    # Load the adversarial model
    device = torch.device(device_str)
    actual_adv_path = adv_model_path.replace(env_name, load_env_name(env_name))
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

    # Collect the data
    noise = torch.distributions.uniform.Uniform(torch.tensor([-1.0], dtype=torch.float32), torch.tensor([1.0], dtype=torch.float32))
    trajs = []
    obs_ = []
    actions_ = []
    rewards_ = []
    infos_ = []
    policy_infos_ = []
    obs, _ = env.reset()
    reward = None
    n_gathered = 0
    t = 0

    with torch.no_grad():
        while n_gathered < n_transitions:
            # Process actions
            policy_infos_.append({})
            obs_.append(copy.deepcopy(obs))
            obs = torch.from_numpy(obs).to(device, dtype=torch.float32)

            pr_action = agent.actor(obs).clamp(-1, 1).cpu()
            if np.random.random() < rnd_prob:
                pr_action = noise.sample(pr_action.shape).view(pr_action.shape).cpu()
            actions_.append(pr_action.numpy())

            adv_action = agent.adversary(obs).clamp(-1, 1).cpu()
            if np.random.random() < rnd_prob:
                adv_action = noise.sample(adv_action.shape).view(adv_action.shape).cpu()
            infos_.append({"adv": adv_action.numpy()})

            # Combine actions and take a step
            comb_action = (pr_action * (1-env_alpha) + adv_action * env_alpha).data.clamp(-1, 1).numpy()
            obs, reward, terminated, truncated, _ = env.step(comb_action)
            done = terminated or truncated
            rewards_.append(reward)
            
            # Process the trajectory if done, and reset
            n_gathered += 1
            t += 1
            if t == traj_length or done:
                trajs.append(Trajectory(
                    obs=obs_,
                    actions=actions_,
                    rewards=rewards_,
                    infos=infos_,
                    policy_infos=policy_infos_
                ))
                t = 0
                obs_ = []
                actions_ = []
                rewards_ = []
                infos_ = []
                policy_infos_ = []
                obs, _ = env.reset()
                reward = None    

    with open(data_path, 'wb') as file:
        pickle.dump(trajs, file)
        print('Saved trajectories to dataset file', data_path)  
    
    return AdvGymEnv(gym.make(load_env_name(env_name)), actual_adv_path, device_str=device_str), trajs
