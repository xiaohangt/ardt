import numpy as np
import torch
from torch.utils.data import IterableDataset

from return_transforms.utils.utils import return_labels


class ARDTDataset(IterableDataset):
    """
    This class provides an iterable dataset for handling trajectories and preparing
    batches for model training of adversarial trajectories.

    Args:
        trajs (list): List of trajectory data.
        n_actions (int): Number of possible actions.
        n_adv_actions (int): Number of adversarial actions.
        horizon (int): The maximum length of a trajectory.
        gamma (float): Discount factor for return calculation. Default is 1.
        act_type (str): Type of actions ('discrete' or 'continuous'). Default is 'discrete'.
        epoch_len (int): Number of iterations (or samples) per epoch. Default is 1e5.
        new_rewards (bool): Whether to use new rewards when calculating returns. Default is False.
    """

    rand: np.random.Generator

    def __init__(
            self, 
            trajs: list, 
            n_actions: int,
            n_adv_actions: int,
            horizon: int,
            gamma: int = 1, 
            act_type: str = 'discrete', 
            epoch_len: float = 1e5, 
            new_rewards: bool = False, 
        ):
        self.trajs = trajs
        self.rets = [return_labels(traj, gamma, new_rewards) for traj in self.trajs]
        self.n_actions = n_actions
        self.n_adv_actions = n_adv_actions
        self.horizon = horizon
        self.act_type = act_type
        self.epoch_len = epoch_len
        self.new_rewards = new_rewards

    def segment_generator(self, epoch_len: int):
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
                # Get actions and adversarial actions for discrete cases
                a = np.array(traj.actions)

                # Check if adversarial actions are present in 'adv' or 'adv_action' key
                if "adv" in traj.infos[0]:
                    adv_a = np.array([info["adv"] if info else -1 for info in traj.infos])
                    adv_actions = np.zeros((adv_a.size, self.n_adv_actions))
                    adv_actions[np.arange(adv_a.size), adv_a] = 1
                    # Handle case where the last step has no adversarial action
                    if traj.infos[-1] == {}:
                        adv_actions[-1] = 0
                elif "adv_action" in traj.infos[0]:
                    adv_a = np.array([info["adv_action"] if info else np.random.randint(self.n_adv_actions) for info in traj.infos])
                    adv_actions = np.zeros((adv_a.size, self.n_adv_actions))
                    adv_actions[np.arange(adv_a.size), adv_a] = 1
                    if traj.infos[-1] == {}:
                        adv_actions[-1] = 0
                else:
                    adv_actions = np.zeros((a.size, self.n_adv_actions))

                # Convert actions to one-hot encoding
                actions = np.zeros((a.size, self.n_actions))
                actions[np.arange(a.size), a] = 1
            else:
                # For continuous actions, no need for one-hot encoding
                actions = np.array(traj.actions)
                adv_actions = np.array([info["adv"] if info else 0. for info in traj.infos])

            # Padding the trajectories to the defined horizon length
            padded_obs = np.zeros((self.horizon, *obs.shape[1:]))
            padded_acts = np.zeros((self.horizon, self.n_actions))
            padded_adv_acts = np.zeros((self.horizon, self.n_adv_actions))
            padded_rets = np.zeros(self.horizon)
            padded_obs[1:obs.shape[0] + 1] = obs
            padded_acts[1:obs.shape[0] + 1] = actions
            padded_adv_acts[1:obs.shape[0] + 1] = adv_actions
            padded_rets[1:obs.shape[0] + 1] = np.array(rets)

            # Yield the padded trajectory segments as tensors
            yield (
                torch.tensor(padded_obs).float(),
                torch.tensor(padded_acts).float(),
                torch.tensor(padded_adv_acts).float(),
                torch.tensor(padded_rets).float()
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
