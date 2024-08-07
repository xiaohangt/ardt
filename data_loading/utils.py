import re
from collections import namedtuple

import numpy as np
import torch

from arrl.ddpg import DDPG
from arrl.main import load_env_name
from arrl.utils import load_model 


Trajectory = namedtuple("Trajectory", ["obs", "actions", "rewards", "infos", "policy_infos"])


class AdvGymEnv:
    def __init__(self, env, adv_model_path, device, env_name, env_alpha=0.1, basic_bm=None):
        self.env = env
        self.env_name = env_name
        self.adv_action_space = env.action_space
        self.adv_model_path = adv_model_path
        self.basic_bm = basic_bm
        self.reset_model_rl(adv_model_path, device)
        self.current_state = None
        self.t = 0
        self.env_alpha = env_alpha

    def reset(self):
        self.t = 0
        state = self.env.reset()
        self.current_state = state
        return state
    
    def reset_model_rl(self, adv_model_path, device):
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
                device=device
            )
            load_model(agent, basedir=adv_model_path)
            self.adv_model = agent

    def step(self, pr_action, adv_action=None):
        if (adv_action is None) and (self.adv_model_path != 'zero') and (self.adv_model is not None):
            state = torch.from_numpy(self.current_state).to(self.adv_model.device, dtype=torch.float32)
            adv_action = self.adv_model.adversary(state).data.clamp(-1, 1).cpu().numpy()
            state, reward, done, _ = self.env.step(pr_action * (1 - self.env_alpha) + adv_action * self.env_alpha)
        else:
            adv_action = np.zeros_like(pr_action)
            state, reward, done, _ = self.env.step(pr_action)
        self.current_state = state
        self.t += 1
        return state, reward, done, {'adv_action': adv_action}

    def __getattr__(self, attr):
        if (attr not in dir(self)) and (attr != 'reset') and (attr != 'step'):
            return self.env.__getattribute__(attr)
        return self.__getattribute__(attr)
