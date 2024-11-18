import numpy as np
import torch
from torch.utils.data import IterableDataset

from return_transforms.utils.utils import return_labels


class ESPERDataset(IterableDataset):
    """
    This class provides an iterable dataset for handling trajectories and preparing
    batches for model training of non-adversarial trajectories.

    Args:
        trajs (list): List of trajectory data.
        n_actions (int): Number of possible actions.
        n_adv_actions (int): Number of adversarial actions.
        horizon (int): The maximum length of a trajectory.
        gamma (float): Discount factor for return calculation. Default is 1.
        act_type (str): Type of actions ('discrete' or 'continuous'). Default is 'discrete'.
        epoch_len (int): Number of iterations (or samples) per epoch. Default is 1e5.
    """
        
    rand: np.random.Generator

    def __init__(
        self, 
        trajs: list, 
        n_actions: int,
        horizon: int,
        gamma: int = 1, 
        act_type: str = 'discrete', 
        epoch_len: float = 1e5
    ):
        self.trajs = trajs
        self.rets = [return_labels(traj, gamma) for traj in self.trajs]
        self.n_actions = n_actions
        self.horizon = horizon
        self.epoch_len = epoch_len
        self.act_type = act_type

    def segment_generator(self, epoch_len):
        """
        Generator function to yield padded segments of trajectory data.

        Args:
            epoch_len (int): Number of samples to generate in this epoch.

        Yields:
            tuple: Padded observations, actions, adversarial actions, and returns.
        """
        for _ in range(epoch_len):
            # Sample a random trajectory
            traj_idx = self.rand.integers(len(self.trajs))
            traj = self.trajs[traj_idx]
            rets = self.rets[traj_idx]
            obs = np.array(traj.obs)
            
            # Handling different action types
            if self.act_type == 'discrete':
                # Convert actions to one-hot encoding
                a = np.array(traj.actions)
                actions = np.zeros((a.size, self.n_actions))
                actions[np.arange(a.size), a] = 1
            else:
                # For continuous actions, no need for one-hot encoding
                actions = np.array(traj.actions)

            # Padding the trajectories to the defined horizon length
            padded_obs = np.zeros((self.horizon, *obs.shape[1:]))
            padded_acts = np.zeros((self.horizon, self.n_actions))
            padded_rets = np.zeros(self.horizon)
            padded_obs[-obs.shape[0]:] = obs
            padded_acts[-obs.shape[0]:] = actions
            padded_rets[-obs.shape[0]:] = np.array(rets)
            true_seq_length = obs.shape[0]

            # Yield the padded trajectory segments as tensors
            yield (
                torch.tensor(padded_obs).float(),
                torch.tensor(padded_acts).float(),
                torch.tensor(padded_rets).float(),
                torch.tensor(true_seq_length).float()
            )

    def __len__(self):
        return int(self.epoch_len)

    def __iter__(self):
        """
        Returns an iterator for the dataset.

        If using multiple workers for data loading, each worker gets a split of the data.
        """
        worker_info = torch.utils.data.get_worker_info()
        self.rand = np.random.default_rng(None)
        
        if worker_info is None:
            # Single-worker setup
            gen = self.segment_generator(int(self.epoch_len))
        else:
            # Multi-worker setup: Split the workload across workers
            per_worker_time_steps = int(self.epoch_len / float(worker_info.num_workers))
            gen = self.segment_generator(per_worker_time_steps)
        
        return gen

