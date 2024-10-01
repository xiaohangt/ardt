import numpy as np
import torch

from data_loading.load_mujoco import Trajectory


def get_past_indices(x: torch.Tensor, seq_len: int) -> torch.Tensor:
    """
    Generate a tensor of indices for non-padded sequences. Assumes padding is applied 
    before the actual sequence.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, T, ...).
        t (int): Length of the sequence.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, T) containing valid indices.
    """
    batch_size = x.size(0)
    obs_seq_len = x.size(1)
    idxs = torch.randint(0, seq_len, (batch_size, obs_seq_len), device=x.device)
    steps = torch.arange(0, seq_len, device=x.device).view(1, obs_seq_len).expand(batch_size, obs_seq_len)

    # Assumes uniform padding length for all sequences in a batch
    # and shifts steps by padding lengths to start from non-padded positions
    pad_lens = obs_seq_len - seq_len
    steps = steps - pad_lens + 1

    # Ensure indices are within valid range and adjust back by padding lengths
    idxs = torch.where(steps == 0, torch.zeros_like(idxs), idxs % steps)
    return (idxs + pad_lens).long()


def return_labels(traj: Trajectory, gamma: float = 1.0, new_rewards: bool = False) -> list:
    """
    Compute the return labels for a trajectory.

    Args:
        traj: Trajectory object
        gamma: Discount factor
        new_rewards: Whether to normalize the returns by the length of the trajectory

    Returns:
        List of return labels
    """
    rewards = traj.rewards
    returns = []
    ret = 0
    for reward in reversed(rewards):
        ret *= gamma
        ret += float(reward)
        if new_rewards:
            returns.append(ret / len(rewards))
        else:
            returns.append(ret)
    returns = list(reversed(returns))
    return returns


def learned_labels(
        traj: Trajectory, label_model: torch.nn.Module, n_actions: int, horizon: int, device: str, act_type: str
    ) -> np.ndarray:
    """
    Compute the learned labels for a trajectory.

    Args:
        traj: Trajectory object
        label_model: torch.nn.Module object
        n_actions: Number of actions
        horizon: Horizon of the model
        device: Device to run the model on
        act_type: Type of actions

    Returns:
        List of learned labels
    """
    with torch.no_grad():
        label_model.eval()
        obs = np.array(traj.obs)
        if act_type == 'discrete':
            a = np.array(traj.actions)
            actions = np.zeros((a.size, n_actions))
            actions[np.arange(a.size), a] = 1
        else:
            actions = np.array(traj.actions)

        padded_obs = np.zeros((horizon, *obs.shape[1:]))
        padded_acts = np.zeros((horizon, n_actions))
        padded_obs[-obs.shape[0]:] = obs
        padded_acts[-obs.shape[0]:] = actions
        padded_obs = torch.tensor(padded_obs).float().unsqueeze(0).to(device)
        padded_acts = torch.tensor(padded_acts).float().unsqueeze(0).to(device)

        labels, _ = label_model.return_preds(padded_obs, padded_acts, hard=True)
        labels = labels[0, -obs.shape[0]:].view(-1).cpu().detach().numpy()

    return np.around(labels, decimals=1)
