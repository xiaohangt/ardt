import time

import numpy as np
import torch
from tqdm import tqdm

from data_loading.load_mujoco import Trajectory


class TrainConfigs:
    def __init__(
        self,
        action_dim: int,
        adv_action_dim: int,
        action_type: str,
        state_dim: int,
        state_mean: float,
        state_std: float,
        returns_scale: int,
        top_pct_traj: float,
        episode_length: int,
        normalize_states: bool = True,
    ):
        self.action_dim = action_dim
        self.adv_action_dim = adv_action_dim
        self.action_type = action_type
        self.state_dim = state_dim
        self.state_mean = state_mean
        self.state_std = state_std
        self.returns_scale = returns_scale
        self.top_pct_traj = top_pct_traj
        self.episode_length = episode_length
        self.normalize_states = normalize_states


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module, 
            optimizer: torch.optim.optimizer,
            scheduler: torch.optim.lr_scheduler,
            gradients_clipper: function | None,
            context_size: int,
            with_adv_action: bool,
            env_name: str,
            trajectories: list[Trajectory],
            trajectories_sorted_idx: np.array[int],
            trajectories_sorted_probs: np.array[float],
            train_configs: TrainConfigs,
            eval_fns: list[function]
        ):
        self.start_time = time.time()
        self.diagnostics = dict()
        # model
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradients_clipper = gradients_clipper
        self.context_size = context_size
        self.with_adv_action = with_adv_action
        # data
        self.env_name = env_name
        self.trajectories = trajectories
        self.trajectories_sorted_idx = trajectories_sorted_idx,
        self.trajectories_sorted_probs = trajectories_sorted_probs
        # training
        self.train_configs = train_configs
        self.loss_fn = self._get_loss_fn(train_configs.action_type)
        # evaluation
        self.eval_fns = [] if eval_fns is None else eval_fns

    def _get_loss_fn(self, action_type):
        if action_type == 'continuous':
            return lambda a_hat, a: torch.mean((a_hat - a)**2)
        else:
            ce_loss = torch.nn.CrossEntropyLoss()
            return lambda a_hat, a: ce_loss(a_hat, torch.argmax(a, dim=-1))

    def get_batch(
            self,  
            device: str = "cpu"
        ):
        # reweights so we sample according to timesteps
        batch_idx = np.random.choice(
            np.arange(len(self.trajectories)),
            size=self.batch_size,
            p=self.trajectories_sorted_probs,
            replace=True,
        )

        # process
        s, a, adv_a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], [], []
        
        for i in range(self.batch_size):
            # selecting trajectories and start point
            traj = self.trajectories[int(self.trajectories_sorted_idx[batch_idx[i]])]
            if self.env_name not in ["halfcheetah", "hopper", "walker2d"]:
                si = 0
            else:
                si = np.random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            if self.env_name == "connect_four":
                cur_state = np.array([obs for obs in traj['observations']])
            else:
                cur_state = traj['observations']
            s.append(cur_state[si:si + self.context_size].reshape(1, -1, self.train_configs.state_dim))
            a.append(traj['actions'][si:si + self.context_size].reshape(1, -1, self.train_configs.act_dim))
            adv_a.append(traj['adv_actions'][si:si + self.context_size].reshape(1, -1, self.train_configs.adv_act_dim))
            r.append(traj['rewards'][si:si + self.context_size].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + self.context_size].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + self.context_size].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.train_configs.episode_length] = self.train_configs.episode_length - 1
            rtg.append(traj['rtg'][si:si + self.context_size].reshape(1, -1, 1))

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, self.context_size - tlen, self.train_configs.state_dim)), s[-1]], axis=1)
            if self.train_configs.normalize_states:
                s[-1] = (s[-1] - self.state_mean) / self.state_std
            a[-1] = np.concatenate([np.ones((1, self.context_size - tlen, self.train_configs.act_dim)) * -10., a[-1]], axis=1)
            adv_a[-1] = np.concatenate([np.zeros((1, self.context_size - tlen, self.train_configs.adv_act_dim)), adv_a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, self.context_size -  tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, self.context_size - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, self.context_size - tlen, 1)),   rtg[-1]], axis=1) / self.train_configs.scale
            timesteps[-1] = np.concatenate([np.zeros((1, self.context_size - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, self.context_size - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        adv_a = torch.from_numpy(np.concatenate(adv_a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        if self.with_adv_action:
            return s, a, adv_a, r, d, rtg, timesteps, mask
        else:
            return s, a, r, d, rtg, timesteps, mask

    def train_iteration(self, num_steps, iter_num=0, device="cpu", print_logs=False):
        # train model
        train_losses = []
        logs = dict()
        train_start = time.time()
        self.model.train()
        for _ in tqdm(range(num_steps)):
            train_loss = self.train_step()
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        # evaluate model
        eval_start = time.time()
        self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/evaluation'] = time.time() - eval_start
        logs['time/total'] = time.time() - train_start

        # log diagonistics metrics
        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs
