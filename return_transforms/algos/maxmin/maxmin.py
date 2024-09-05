import gym
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.autonotebook import tqdm

from data_loading.load_mujoco import Trajectory
from return_transforms.models.ardt.maxmin_model import RtgFFN, RtgLSTM
from return_transforms.datasets.ardt_dataset import ARDTDataset
from return_transforms.datasets.discretizer import TrajectoryDiscretizer


def _expectile_fn(
        td_error: torch.Tensor, 
        acts_mask: torch.Tensor, 
        alpha: float = 0.01, 
        discount_weighted: bool = False
    ):
    batch_loss = torch.abs(alpha - F.normalize(F.relu(td_error), dim=-1)) 
    batch_loss *= (td_error ** 2)
    if discount_weighted:
        weights = 0.5 ** np.array(range(len(batch_loss)))[::-1]
        return (
            batch_loss[~acts_mask] * torch.from_numpy(weights).to(td_error.device)
        ).mean()
    else:
        return (batch_loss.squeeze(-1) * ~acts_mask).mean()
    

def maxmin(
        trajs: list[Trajectory],
        action_space: gym.spaces,
        adv_action_space: gym.spaces,
        train_args: dict,
        device: str,
        n_cpu: int,
        is_simple_model: bool = False,
        is_toy: bool = False,
        is_discretize: bool = False,
    ):
    # Discretize dataset if so specified (currently always set to false)
    if is_discretize:
        discretizer = TrajectoryDiscretizer(trajs, 12, 20)
        action_type = 'discrete'
        action_size = discretizer.discrete_acts_dim
        adv_action_size = discretizer.discrete_adv_acts_dim
        data_trajs = discretizer.discrete_traj
    else:
        data_trajs = trajs

    # Initialize state and action spaces
    obs_size = np.prod(trajs[0].obs[0].shape)
    if isinstance(action_space, gym.spaces.Discrete):
        action_size = action_space.n
        adv_action_size = adv_action_space.n
        action_type = 'discrete'
    else:
        action_size = action_space.shape[0]
        adv_action_size = adv_action_space.shape[0]
        action_type = 'continuous'

    # Build dataset and dataloader
    max_len = max([len(traj.obs) for traj in trajs]) + 1
    dataset = ARDTDataset(
        data_trajs, 
        action_size, 
        adv_action_size, 
        max_len, 
        gamma=train_args['gamma'], 
        act_type=action_type
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=train_args['batch_size'], num_workers=n_cpu
    )

    # Set up the models
    print(f'Creating models... (simple={is_simple_model})')
    if is_simple_model:
        qsa_pr_model = RtgFFN(obs_size, action_size, include_adv=False).to(device)
        qsa_adv_model = RtgFFN(obs_size, action_size, adv_action_size, include_adv=True).to(device)
    else:
        qsa_pr_model = RtgLSTM(obs_size, action_size, adv_action_size, train_args, include_adv=False).to(device)
        qsa_adv_model = RtgLSTM(obs_size, action_size, adv_action_size, train_args, include_adv=True).to(device)

    qsa_pr_optimizer = torch.optim.AdamW(
        qsa_pr_model.parameters(), lr=train_args['model_lr'], weight_decay=train_args['model_wd']
    )
    qsa_adv_optimizer = torch.optim.AdamW(
        qsa_adv_model.parameters(), lr=train_args['model_lr'], weight_decay=train_args['model_wd']
    )

    # Start training
    mse_epochs = train_args['mse_epochs']
    maxmin_epochs = train_args['maxmin_epochs'] 
    total_epochs = mse_epochs + maxmin_epochs
    assert maxmin_epochs % 2 == 0

    print('Training...')
    qsa_pr_model.train()
    qsa_adv_model.train()
    
    for epoch in range(total_epochs):
        pbar = tqdm(dataloader, total=len(dataloader))
        total_loss = 0
        total_pr_loss = 0
        total_adv_loss = 0
        total_batches = 0

        for obs, acts, adv_acts, ret in pbar:
            total_batches += 1
            qsa_pr_optimizer.zero_grad()
            qsa_adv_optimizer.zero_grad()
            
            # Set up variables
            batch_size = obs.shape[0]
            seq_len = obs.shape[1]
            
            obs = obs.view(batch_size, seq_len, -1).to(device)
            acts = acts.to(device)
            adv_acts = adv_acts.to(device)
            acts_mask = (acts.sum(dim=-1) == 0)
            ret = (ret / train_args['scale']).to(device)

            # Adjustment for initial prompt learning
            obs[:, 0] = obs[:, 1]
            ret[:, 0] = ret[:, 1]
            acts_mask[:, 0] = False

            # Adjustment for toy environment
            if is_toy:
                obs, acts, adv_acts, acts_mask, ret = (
                    obs[:, :-1], acts[:, :-1], adv_acts[:, :-1], acts_mask[:, :-1], ret[:, :-1]
                )
                seq_len -= 1

            # Calculate the losses at the different tages
            if epoch < mse_epochs:
                # MSE to learn general loss landscape
                ret_pr_pred = qsa_pr_model(obs, acts).view(batch_size, seq_len)
                ret_pr_loss = (((ret_pr_pred - ret) ** 2) * ~acts_mask).mean()
                ret_adv_pred = qsa_adv_model(obs, acts, adv_acts).view(batch_size, seq_len)
                ret_adv_loss = (((ret_adv_pred - ret) ** 2) * ~acts_mask).mean()
                # Backpropagate
                ret_pr_loss.backward()
                qsa_pr_optimizer.step()
                ret_adv_loss.backward()
                qsa_adv_optimizer.step()
                # Update
                total_loss += ret_pr_loss.item() + ret_adv_loss.item()
                total_pr_loss += ret_pr_loss.item()
                total_adv_loss += ret_adv_loss.item()
            elif epoch % 2 == 0:
                # Max step: protagonist attempts to maximise at each node
                ret_pr_pred = qsa_pr_model(obs, acts)
                ret_adv_pred = qsa_adv_model(obs, acts, adv_acts)
                ret_pr_loss = _expectile_fn(ret_pr_pred - ret_adv_pred.detach(), acts_mask, train_args['alpha'])            
                # Backpropagate
                ret_pr_loss.backward()
                qsa_pr_optimizer.step()
                # Update 
                total_loss += ret_pr_loss.item()
                total_pr_loss += ret_pr_loss.item()
            else:
                # Min step: adversary attempts to minimise at each node             
                rewards = (ret[:, :-1] - ret[:, 1:]).view(batch_size, -1, 1)
                ret_pr_pred = qsa_pr_model(obs, acts)
                ret_adv_pred = qsa_adv_model(obs, acts, adv_acts)
                ret_tree_loss = _expectile_fn(
                    ret_pr_pred[:, 1:].detach() + rewards - ret_adv_pred[:, :-1], 
                    acts_mask[:, :-1], 
                    train_args['alpha']
                )
                ret_leaf_loss = (
                    (ret_adv_pred[range(batch_size), -1].flatten() - ret[range(batch_size), -1]) ** 2
                ).mean()
                ret_adv_loss = ret_tree_loss * (1 - train_args['leaf_weight']) + ret_leaf_loss * train_args['leaf_weight']                    
                # Backpropagate
                ret_adv_loss.backward()
                qsa_adv_optimizer.step()
                # Update
                total_loss += ret_adv_loss.item()
                total_adv_loss += ret_adv_loss.item()

            pbar.set_description(
                f"Epoch {epoch} | Total Loss: {total_loss / total_batches:.4f} | Pr Loss: {total_pr_loss / total_batches:.4f} | Adv Loss: {total_adv_loss / total_batches:.4f}"
            )

    # Get the learned return labels and prompt values (i.e. highest returns-to-go)
    with torch.no_grad():
        learned_returns = []
        prompt_value = -np.inf    
        for traj in tqdm(data_trajs):
            # Predict returns
            obs = torch.from_numpy(np.array(traj.obs)).float().to(device).view(1, -1, obs_size)
            acts = torch.from_numpy(np.array(traj.actions)).to(device).view(1, -1)
            if action_type == "discrete" and not is_discretize:
                acts = torch.nn.functional.one_hot(acts, num_classes=action_size)
            else:
                acts = acts.view(1, -1, action_size)
            returns = qsa_pr_model(
                obs.view(obs.shape[0], -1, obs_size), acts.float()
            ).cpu().flatten().numpy()
            # Compare against previously held prompt value
            if prompt_value < returns[-len(traj.actions)]:
                prompt_value = returns[-len(traj.actions)]
            # Update
            learned_returns.append(np.round(returns * train_args['scale'], decimals=3))

    return learned_returns, np.round(prompt_value * train_args['scale'], decimals=3)
