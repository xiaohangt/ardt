from return_transforms.models.esper.cluster_model import ClusterModel
from return_transforms.models.esper.dynamics_model import DynamicsModel
from return_transforms.datasets.esper_dataset import ESPERDataset
from return_transforms.utils.utils import learned_labels
from tqdm.autonotebook import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym


def esper(trajs,
          action_space,
          dynamics_model_args,
          cluster_model_args,
          train_args,
          device,
          n_cpu):

    # Check if discrete action space
    if isinstance(action_space, gym.spaces.Discrete):
        action_size = action_space.n
        act_loss_fn = lambda pred, truth: F.cross_entropy(pred.view(-1, pred.shape[-1]), torch.argmax(truth, dim=-1).view(-1),
                                                          reduction='none')
        act_type = 'discrete'
    else:
        action_size = action_space.shape[0]
        act_loss_fn = lambda pred, truth: ((pred - truth) ** 2).mean(dim=-1)
        act_type = 'continuous'

    # Get the length of the longest trajectory
    max_len = max([len(traj.obs) for traj in trajs]) + 1

    dataset = ESPERDataset(trajs, action_size, max_len,
                           gamma=train_args['gamma'], act_type=act_type)

    scale = train_args['scale']

    # Get the obs size from the first datapoint
    obs, _, _, _ = next(iter(dataset))
    obs_shape = obs[0].shape
    obs_size = np.prod(obs_shape)

    # Set up the models
    print('Creating models...')
    dynamics_model = DynamicsModel(obs_size,
                                   action_size,
                                   cluster_model_args['rep_size'],
                                   dynamics_model_args).to(device)

    cluster_model = ClusterModel(obs_size,
                                 action_size,
                                 cluster_model_args['rep_size'],
                                 cluster_model_args,
                                 cluster_model_args['groups']).to(device)

    dynamics_optimizer = optim.AdamW(
        dynamics_model.parameters(), lr=float(train_args['dynamics_model_lr']))
    cluster_optimizer = optim.AdamW(
        cluster_model.parameters(), lr=float(train_args['cluster_model_lr']))

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=train_args['batch_size'],
                                             num_workers=n_cpu)

    # Calculate epoch markers
    total_epochs = train_args['cluster_epochs'] + train_args['return_epochs']
    ret_stage = train_args['cluster_epochs']

    print('Training...')

    dynamics_model.train()
    cluster_model.train()
    for epoch in range(total_epochs):
        pbar = tqdm(dataloader, total=len(dataloader))
        total_loss = 0
        total_act_loss = 0
        total_ret_loss = 0
        total_dyn_loss = 0
        total_baseline_dyn_loss = 0
        total_batches = 0
        for obs, acts, ret, seq_len in pbar:
            total_batches += 1
            # Take an optimization step for the cluster model
            cluster_optimizer.zero_grad()
            obs = obs.to(device)
            acts = acts.to(device)
            ret = ret.to(device) / scale
            seq_len = seq_len.to(device)

            bsz, t = obs.shape[:2]

            act_mask = (acts.sum(dim=-1) == 0)
            obs_mask = (obs.view(bsz, t, -1)[:, :-1].sum(dim=-1) == 0)

            # Get the cluster predictions
            clusters, ret_pred, act_pred, _ = cluster_model(
                obs, acts, seq_len, hard=epoch >= ret_stage)

            pred_next_obs, next_obs = dynamics_model(
                obs, acts, clusters, seq_len)

            # Calculate the losses

            ret_loss = ((ret_pred.view(bsz, t) - ret.view(bsz, t)) ** 2).mean()
            act_loss = act_loss_fn(act_pred, acts).view(bsz, t)[
                ~act_mask].mean()
            dynamics_loss = ((pred_next_obs - next_obs) ** 2)[~obs_mask].mean()

            # Calculate the total loss
            if epoch < ret_stage:
                loss = -train_args['adv_loss_weight'] * dynamics_loss + \
                    train_args['act_loss_weight'] * act_loss
            else:
                loss = ret_loss

            loss.backward()
            cluster_optimizer.step()

            # Take an optimization step for the dynamics model
            dynamics_optimizer.zero_grad()
            pred_next_obs, next_obs = dynamics_model(
                obs, acts, clusters.detach(), seq_len)
            baseline_pred_next_obs, _ = dynamics_model(
                obs, acts, torch.zeros_like(clusters), seq_len)
            dynamics_loss = ((pred_next_obs - next_obs) ** 2)[~obs_mask].mean()
            baseline_dynamics_loss = (
                (baseline_pred_next_obs - next_obs) ** 2)[~obs_mask].mean()
            total_dynamics_loss = dynamics_loss + baseline_dynamics_loss
            total_dynamics_loss.backward()
            dynamics_optimizer.step()

            # Update the progress bar
            total_loss += loss.item()
            total_act_loss += act_loss.item()
            total_ret_loss += ret_loss.item()
            total_dyn_loss += dynamics_loss.item()
            total_baseline_dyn_loss += baseline_dynamics_loss.item()

            advantage = total_baseline_dyn_loss - total_dyn_loss

            pbar.set_description(
                f"Epoch {epoch} | Loss: {total_loss / total_batches:.4f} | Act Loss: {total_act_loss / total_batches:.4f} | Ret Loss: {total_ret_loss / total_batches:.4f} | Dyn Loss: {total_dyn_loss / total_batches:.4f} | Adv: {advantage / total_batches:.4f}")

    # test_env(cluster_model, action_size, max_len, device, act_type)

    # Get the learned return labels
    avg_returns = []
    for traj in tqdm(trajs):
        labels = learned_labels(traj, cluster_model,
                                action_size, max_len, device, act_type)
        avg_returns.append(labels * scale)

    return avg_returns


def test_env(label_model, n_actions, horizon, device, act_type='discrete'):
    from stochastic_offline_envs.envs.offline_envs.connect_four_offline_env import ConnectFourOfflineEnv
    from stochastic_offline_envs.envs.connect_four.connect_four_env import GridWrapper
    task = ConnectFourOfflineEnv(worst_case_adv=True, data_name="c4data_mdp_20", test_only=True)
    env = GridWrapper(task.test_env_cls())
    reward_list = []

    with torch.no_grad():
        for _ in range(100):
            state, _ = env.reset()

            padded_obs = torch.zeros([1, 1, 84]).to(device).float()
            padded_acts = torch.zeros([1, 1, 7]).to(device).float()

            for _ in range(23):
                q_values = []

                for i in range(7):
                    obs = torch.tensor(state).to(device).float().view(1, 1, -1)
                    one_hot_act = torch.nn.functional.one_hot(torch.tensor(i).to(device), num_classes=7).float().view(1, 1, -1)

                    padded_obs = torch.cat([padded_obs, obs], dim=1)
                    padded_acts = torch.cat([padded_acts, one_hot_act], dim=1)

                    labels, _ = label_model.return_preds(padded_obs, padded_acts, hard=True)
                    q_values.append(labels.flatten()[-1].cpu().numpy())

                action = np.argmax(q_values)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                if done:
                    break

            reward_list.append(reward)
        print(np.mean(reward_list), np.std(reward_list))
