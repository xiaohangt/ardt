from return_transforms.models.esper.cluster_model import ClusterModel
from return_transforms.models.esper.dynamics_model import DynamicsModel
from return_transforms.models.ardt.maxmin_model import RtgNetwork, AdvPolicyNetwork, NewRtgNetwork
from return_transforms.datasets.esper_dataset import ESPERDataset
from return_transforms.datasets.ardt_dataset import ARDTDataset
from return_transforms.datasets.discretizer import TrajectoryDiscretizer

from return_transforms.utils.utils import learned_labels
from tqdm.autonotebook import tqdm
from copy import deepcopy

import gc
import pickle
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def worst_case_qf(
          env_name,
          trajs,
          action_space,
          adv_action_space,
          train_args,
          device,
          n_cpu,
          lr,
          wd,
          is_old_model,
          batch_size,
          leaf_weight=0.5,
          alpha=0.01,
          discretization=False):

    # Check if discrete action space
    if isinstance(action_space, gym.spaces.Discrete):
        action_size = action_space.n
        adv_action_size = adv_action_space.n
        act_type = 'discrete'
    else:
        action_size = action_space.shape[0]
        adv_action_size = adv_action_space.shape[0]
        act_type = 'continuous'

    # Get the length of the longest trajectory
    max_len = max([len(traj.obs) for traj in trajs]) + 1
    if discretization:
        discretizer = TrajectoryDiscretizer(trajs, 12, 20) # discretizer.discrete_traj
        act_type = 'discrete'
        action_size = discretizer.discrete_acts_dim
        adv_action_size = discretizer.discrete_adv_acts_dim
        data_trajs = discretizer.discrete_traj
    else:
        data_trajs = trajs

    dataset = ARDTDataset(data_trajs, action_size, adv_action_size, max_len, gamma=train_args['gamma'], act_type=act_type)

    scale = train_args['scale']

    # Get the obs size from the first datapoint
    obs, _, _, _, _ = next(iter(dataset))

    obs_shape = obs[0].shape
    obs_size = np.prod(obs_shape)

    # Set up the models
    print(f'Creating models... It\'s old models: {is_old_model}')
    if is_old_model:
        qsa2_model = RtgNetwork(obs_size, action_size, adv_action_size, include_adv=True).to(device)
        qsa_model = RtgNetwork(obs_size, action_size, include_adv=False).to(device)
    else:
        qsa2_model = NewRtgNetwork(obs_size, action_size, adv_action_size, train_args, include_adv=True).to(device)
        qsa_model = NewRtgNetwork(obs_size, action_size, adv_action_size, train_args, include_adv=False).to(device)


    qsa2_optimizer = optim.AdamW(qsa2_model.parameters(), lr=lr, weight_decay=wd)
    qsa_optimizer = optim.AdamW(qsa_model.parameters(), lr=lr, weight_decay=wd)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=n_cpu)

    # Calculate epoch markers
    total_epochs = train_args['mse_epochs'] + train_args['maxmin_epochs'] 
    ret_stage = train_args['maxmin_epochs']
    assert train_args['maxmin_epochs'] % 2 == 0

    print('Training...')
    qsa2_model.train()
    qsa_model.train()

    def expectile_fn(td_error, act_mask, discount_weighted=False):
        batch_loss = torch.abs(alpha - F.normalize(F.relu(td_error), dim=-1)) * (td_error ** 2)
        assert not discount_weighted
        if discount_weighted:
            weights = 0.5 ** np.array(range(len(batch_loss)))[::-1]
            return (batch_loss[~act_mask] * torch.from_numpy(weights).to(td_error.device)).mean()
        else:
            return (batch_loss.squeeze(-1) * ~act_mask).mean()
    
    for epoch in range(total_epochs):
        pbar = tqdm(dataloader, total=len(dataloader))
        total_loss = 0
        total_act_loss = 0
        total_ret_loss = 0
        total_batches = 0

        for obs, acts, ret, seq_len, adv_acts in pbar:
            total_batches += 1
            bsz, t = obs.shape[:2]
            # Recover adv
            if env_name == "toy":
                t -= 1
                # adv_acts_ind = (torch.where(obs[:, -1] > 0)[1] - 1) % 3
                # adv_acts = torch.nn.functional.one_hot(adv_acts_ind).unsqueeze(1).float().to(device) # (bsz, seq_len-1, ...)
                obs, acts, ret, adv_acts = obs[:, :-1], acts[:, :-1], ret[:, :-1], adv_acts[:, :-1]

            # some environment has no terminal state
            if seq_len.max() >= t:
                seq_len -= 1

            # Take an optimization step for the cluster model
            qsa2_optimizer.zero_grad()
            qsa_optimizer.zero_grad()
            obs = obs.view(bsz, t, -1).to(device)
            acts = acts.to(device)
            adv_acts = adv_acts.to(device)
            ret = ret.to(device) / scale
            seq_len = seq_len.to(device)

            # Calculate the losses
            loss, act_loss, ret_loss = torch.tensor(0), torch.tensor(0), torch.tensor(0)

            act_mask = (acts.sum(dim=-1) == 0) 
            assert act_mask.shape == ret.shape
            # for initial prompt learning
            obs[:, 0] = obs[:, 1]
            ret[:, 0] = ret[:, 1]
            act_mask[:, 0] = False 
            
            # Calculate the total loss
            if epoch < ret_stage:              
                ret_a2_pred = qsa2_model(obs, acts, adv_acts).view(bsz, t)
                ret_a_pred = qsa_model(obs, acts).view(bsz, t)
                ret_a2_loss = (((ret_a2_pred - ret) ** 2) * ~act_mask).mean()
                ret_a_loss = (((ret_a_pred - ret) ** 2) * ~act_mask).mean()
                ret_loss = ret_a_loss + ret_a2_loss

                ret_a2_loss.backward()
                qsa2_optimizer.step()
                ret_a_loss.backward()
                qsa_optimizer.step()

                total_loss += ret_loss.item() + act_loss.item()
                total_ret_loss += ret_a_loss.item()
                total_act_loss += ret_a2_loss.item()
            elif epoch % 2 == 0:
                ret_a2_pred = qsa2_model(obs, acts, adv_acts)
                ret_a_pred = qsa_model(obs, acts)
                ret_a_loss = expectile_fn(ret_a_pred - ret_a2_pred.detach(), act_mask)            
                ret_loss = ret_a_loss 
                ret_a_loss.backward()
                qsa_optimizer.step()       
            else:                
                rewards = (ret[:, :-1] - ret[:, 1:]).view(bsz, -1, 1)
                ret_a_pred = qsa_model(obs, acts)
                ret_a2_pred = qsa2_model(obs, acts, adv_acts)

                ret_leaf_loss = ((ret_a2_pred[range(bsz), seq_len].flatten() - ret[range(bsz), seq_len]) ** 2).mean()
                ret_a2_loss = expectile_fn(ret_a_pred[:, 1:].detach() + rewards - ret_a2_pred[:, :-1], act_mask[:, :-1]) * (1 - leaf_weight) + ret_leaf_loss * leaf_weight                    

                ret_loss = ret_a2_loss
                ret_a2_loss.backward()
                qsa2_optimizer.step()

                total_loss += ret_loss.item() + act_loss.item()
                total_act_loss += act_loss.item()
                total_ret_loss += ret_loss.item()
                
            pbar.set_description(
                f"Epoch {epoch} | Loss: {total_loss / total_batches:.4f} | loss 1: {total_ret_loss / total_batches:.4f} | loss 2: {total_act_loss / total_batches:.4f}")



    # Get the learned return labels
    with torch.no_grad():
        avg_returns, trajs_prune_labels, prompt_value = [], [], -np.inf
        for traj in tqdm(data_trajs):
            seq_len = len(traj.actions)
            obs = torch.from_numpy(np.array(traj.obs)).float().to(device).view(1, -1, obs_size)
            acts = torch.from_numpy(np.array(traj.actions)).to(device).view(1, -1)
            if act_type == "discrete" and not discretization:
                acts = torch.nn.functional.one_hot(acts, num_classes=action_size)
            else:
                acts = acts.view(1, -1, action_size)
            returns = qsa_model(obs.view(obs.shape[0], -1, obs_size), acts.float()).cpu().flatten().numpy()
            if prompt_value < returns[-seq_len]:
                prompt_value = returns[-seq_len]

            # ret_a2_pred = qsa2_model(obs.view(obs.shape[0], -1, obs_size), acts.float(), adv_a.float()).cpu().flatten().numpy()

            avg_returns.append(np.round(returns * scale, decimals=3))

    return avg_returns, np.round(prompt_value * scale, decimals=3), qsa2_model
    
def get_one_hot(predicted_adv):
    predicted_adv_top2 =  torch.topk(predicted_adv, k=2, dim=-1)[0]
    threshold = predicted_adv_top2.mean(dim=-1).unsqueeze(dim=-1)
    one_hot_predicted_adv = torch.nn.functional.relu(predicted_adv - threshold) 
    return torch.nn.functional.normalize(one_hot_predicted_adv)

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False