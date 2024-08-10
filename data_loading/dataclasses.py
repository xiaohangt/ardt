import re
from collections import namedtuple

import gym
import numpy as np
import torch

from arrl.ddpg import DDPG
from arrl.main import load_env_name
from arrl.utils import load_model 


Trajectory = namedtuple("Trajectory", ["obs", "actions", "rewards", "infos", "policy_infos"])


class AdvGymEnv:
    """
    A container class for a gym environment that includes an adversary model.

    Args:
        env (gym.Env): The environment.
        adv_model_path (str): The path to the adversarial model.
        basic_bm (float): The basic body mass.
        env_alpha (float): The environment alpha.
        device_str (str): The device to use.
    """
    def __init__(
            self, 
            env: gym.Env,
            env_name: str,
            adv_model_path: str,  
            basic_bm: float = None,
            env_alpha: float = 0.1, 
            device_str: str = "cpu",
        ):
        self.env = env
        self.env_name = env_name
        self.adv_model_path = adv_model_path
        self.adv_action_space = env.action_space  # same space in action-robustness framework
        self.basic_bm = basic_bm
        self.env_alpha = env_alpha
        self.reset_adv_agent(adv_model_path, device_str)
        self.current_state = None
        self.t = 0

    def reset(self) -> np.ndarray:
        self.t = 0
        self.current_state, _ = self.env.reset() 
        return self.current_state, None
    
    def reset_adv_agent(self, adv_model_path: str, device_str: str) -> None:   
        """
        Reset the adversary agent in the environment.

        Args:
            adv_model_path (str): The path to the adversary model.
            device_str (str): The device
        """
        print("Reset adversary:", adv_model_path)
        self.adv_model = None
        if 'env' in adv_model_path:
            mass = eval(re.findall(r'\d+\.\d+|\d+', adv_model_path)[0])
            for idx in range(len(self.basic_bm)):
                self.env.env.env.model.body_mass[idx] = self.basic_bm[idx] * mass
        elif adv_model_path != 'zero':
            adv_model_path = adv_model_path.replace(self.env_name, load_env_name(self.env_name))
            agent = DDPG(
                gamma=1, 
                tau=1, 
                hidden_size=64, 
                num_inputs=self.env.observation_space.shape[0],
                action_space=self.env.action_space.shape[0], 
                train_mode=False, 
                alpha=1, 
                replay_size=1, 
                device=torch.device(device_str)
            )
            load_model(agent, basedir=adv_model_path)
            self.adv_model = agent

    def step(
            self, 
            pr_action: np.ndarray,
            adv_action: np.ndarray | None = None
        ) -> tuple[np.ndarray, float, bool, dict]:
        """
        Take a step in the environment.

        Args:
            pr_action (np.ndarray): The protagonist action.
            adv_action (np.ndarray): The adversary action.

        Returns:
            state (np.ndarray): The new state.
            reward (float): The reward.
            done (bool): The done flag.
            dict: The dictionary of the adversarial action taken.
        """
        alpha = self.env_alpha
        if adv_action is None and self.adv_model_path != 'zero' and self.adv_model is not None:
            # Adversary action generated using the adversary agent
            state = torch.from_numpy(self.current_state).to(self.adv_model.device, dtype=torch.float32)
            adv_action = self.adv_model.adversary(state).data.clamp(-1, 1).cpu().numpy()
        elif adv_action is None:
            # No adversary action
            alpha = 0.0
            adv_action = np.zeros_like(pr_action)
        comb_action = pr_action * (1 - alpha) + adv_action * alpha
        state, reward, terminated, truncated, _ = self.env.step(comb_action)
        done = terminated or truncated
        self.current_state = state
        self.t += 1
        return state, reward, done, None, {'adv_action': adv_action}

    def __getattr__(self, attr):
        if (attr not in dir(self)) and (attr != 'reset') and (attr != 'step'):
            return self.env.__getattribute__(attr)
        return self.__getattribute__(attr)
