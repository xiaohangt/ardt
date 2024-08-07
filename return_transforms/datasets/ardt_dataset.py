import numpy as np
import torch
from torch.utils.data import IterableDataset

from return_transforms.utils.utils import return_labels


class ARDTDataset(IterableDataset):
    rand: np.random.Generator

    def __init__(
            self, 
            trajs, 
            n_actions, 
            n_adv_actions, 
            horizon, 
            gamma=1, 
            act_type='discrete', 
            epoch_len=1e5, 
            new_rewards=False, 
            discretizer=None
        ):
        self.trajs = trajs
        self.rets = [return_labels(traj, gamma, new_rewards) for traj in self.trajs]
        self.n_actions = n_actions
        self.n_adv_actions = n_adv_actions
        self.horizon = horizon
        self.epoch_len = epoch_len
        self.act_type = act_type
        self.new_rewards = new_rewards

    def segment_generator(self, epoch_len):
        for _ in range(epoch_len):
            traj_idx = self.rand.integers(len(self.trajs))
            traj = self.trajs[traj_idx]
            rets = self.rets[traj_idx]
            if self.new_rewards and len(np.where(np.array(rets) > 1)[0]) > 0:
                breakpoint()
            if self.act_type == 'discrete':
                a = np.array(traj.actions)
                if "adv" in traj.infos[0]:
                    adv_a = np.array([info["adv"] if info else -1 for info in traj.infos])
                    adv_actions = np.zeros((adv_a.size, self.n_adv_actions))
                    adv_actions[np.arange(adv_a.size), adv_a] = 1
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
                actions = np.zeros((a.size, self.n_actions))
                actions[np.arange(a.size), a] = 1
            else:
                actions = np.array(traj.actions)
                adv_actions = np.array([info["adv"] if info else 0. for info in traj.infos])
            obs = np.array(traj.obs)

            padded_obs = np.zeros((self.horizon, *obs.shape[1:]))
            padded_acts = np.zeros((self.horizon, self.n_actions))
            padded_adv_acts = np.zeros((self.horizon, self.n_adv_actions))
            padded_rets = np.zeros(self.horizon)

            padded_obs[1:obs.shape[0] + 1] = obs
            padded_acts[1:obs.shape[0] + 1] = actions
            padded_adv_acts[1:obs.shape[0] + 1] = adv_actions
            padded_rets[1:obs.shape[0] + 1] = np.array(rets)
            seq_length = obs.shape[0]

            yield torch.tensor(padded_obs).float(), \
                torch.tensor(padded_acts).float(), \
                torch.tensor(padded_rets).float(), \
                torch.tensor(seq_length).long(), \
                torch.tensor(padded_adv_acts).float()

    def __len__(self):
        return int(self.epoch_len)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        self.rand = np.random.default_rng(None)
        if worker_info is None:  # single-process data loading, return the full iterator
            gen = self.segment_generator(int(self.epoch_len))
        else:  # in a worker process
            # split workload
            per_worker_time_steps = int(
                self.epoch_len / float(worker_info.num_workers))
            gen = self.segment_generator(per_worker_time_steps)
        return gen
