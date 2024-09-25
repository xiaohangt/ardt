import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

from stochastic_offline_envs.envs.connect_four.connect_four_env import GridWrapper
from stochastic_offline_envs.stochastic_offline_envs.envs.offline_envs.base import BaseOfflineEnv


def worst_case_env_step(
        state: np.array, 
        action: np.array, 
        timestep: int,
        env_name: str, 
        env: BaseOfflineEnv
    ):
    # Hardcoded class for worst-case adversaries in toy environments
    new_state_ind = -1
    adv_action = np.random.choice(2, 1)
    
    _, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    
    if env_name == "GamblingEnv":
        if timestep == 0:
            if action == 0:
                w_reward = -15 
            elif action == 1:
                w_reward = -6
            else:
                w_reward = 1
        elif timestep == 1:
            reward = w_reward
            assert done
    elif env_name == "ToyEnv":
        if timestep == 0:
            if action == 0:
                env.w_reward = 0
            else:
                env.w_reward = 1
        else:
            reward = env.w_reward
            assert done
    elif env_name == "MSToyEnv":
        done = False
        if timestep == 0:
            if action > 0:
                reward = 4
                done = True
            else:
                new_state_ind = 1
                reward = 0
                adv_action = 0
        else:
            reward = env.reward_list[action + (state.argmax() - 1) * 3]
            done = True
    elif env_name == "NewMSToyEnv":
        done = False
        if timestep == 0:
            if action > 0:
                reward = 4
                done = True
            else:
                new_state_ind = 1
                reward = 0
        else:
            reward = env.reward_list[action * 2 + (state.argmax() - 1) * 3]
            done = True
    else:
        raise RuntimeError("Environment Error.")

    new_state = np.eye(state.size)[new_state_ind] if new_state_ind != -1 else state
    return new_state, reward, done, False, {"adv_action": adv_action}


def evaluate_episode(
        env,
        env_name,
        state_dim,
        act_dim,
        action_type,
        model,
        model_type,
        max_ep_len,
        scale,
        state_mean,
        state_std,
        target_return,
        adv_act_dim=None,
        normalize_states=False,
        worst_case=True,
        with_noise=False,
        device='cpu',
    ):
    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state, _ = env.reset()
    if with_noise:
        state = state + np.random.normal(0, 0.1, size=state.shape)

    if not adv_act_dim:
        adv_act_dim = act_dim

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    adv_actions = torch.zeros((0, adv_act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32).reshape(1, 1)

    # evaluate
    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):
        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        adv_actions = torch.cat([adv_actions, torch.zeros((1, adv_act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        
        if normalize_states:
            normalized_states = (states.to(dtype=torch.float32) - state_mean) / state_std
        else:
            normalized_states = states.to(dtype=torch.float32)
        
        if model_type == 'dt':
            action = model.get_action(
                states=normalized_states.to(dtype=torch.float32), 
                actions=actions.to(dtype=torch.float32),
                rewards=rewards.to(dtype=torch.float32),
                returns_to_go=target_return.to(dtype=torch.float32),
                timesteps=timesteps.to(dtype=torch.long),
                batch_size=1
            )
            action = action[0, -1]
        elif model_type == 'adt':
            action = model.get_action(
                states=normalized_states.to(dtype=torch.float32), 
                actions=actions.to(dtype=torch.float32),
                adv_actions=adv_actions.to(dtype=torch.float32),
                rewards=rewards.to(dtype=torch.float32),
                returns_to_go=target_return.to(dtype=torch.float32),
                timesteps=timesteps.to(dtype=torch.long),
                batch_size=1
            )
            action = action[0, -1]
        elif model_type == "bc":
            action = model.get_action(
                states=normalized_states.to(dtype=torch.float32), 
                actions=actions.to(dtype=torch.float32),
                rewards=rewards.to(dtype=torch.float32),
            )

        if action_type == 'discrete':
            # sample action
            act_probs = F.softmax(action, dim=-1)
            action = Categorical(probs=act_probs).sample()
            # make the action one hot
            one_hot_action = torch.zeros(1, act_dim).float()
            one_hot_action[0, action] = 1
            actions[-1] = one_hot_action
        else:
            actions[-1] = action
        
        action = action.detach().cpu().numpy()

        if (
            worst_case and 
            env_name in ["GamblingEnv", "ToyEnv", "MSToyEnv", "NewMSToyEnv"]
        ):
            state, reward, terminated, truncated, infos = worst_case_env_step(
                state, action, t, env, env_name
            )
        else:
            state, reward, terminated, truncated, infos = env.step(action)
        done = terminated or truncated

        if action_type == 'discrete':
            one_hot_adv_action = torch.zeros(1, adv_act_dim).float()
            if infos != {}:
                adv_a = infos["adv"] if "adv" in infos else infos["adv_action"]
                one_hot_adv_action[0, adv_a] = 1
            adv_actions[-1] = one_hot_adv_action
        else:
            adv_action = infos["adv"] if "adv" in infos else infos["adv_action"]
            adv_actions[-1] = torch.from_numpy(adv_action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        pred_return = target_return[0,-1] - (reward/scale)
        
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1
        )
        
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1
        )

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def evaluate(
        task, 
        env_name,
        num_eval_episodes,
        state_dim, 
        act_dim, 
        adv_act_dim,
        action_type,
        model, 
        model_type,
        max_ep_len, 
        scale, 
        state_mean, 
        state_std,
        target_return,
        batch_size=1, 
        normalize_states=True,
        device='cpu'
    ):
    test_env = task.test_env_cls()
    if env_name == "ConnectFourEnv":
        test_env = GridWrapper(test_env) 

    returns, lengths = [], []
    for _ in tqdm(range(num_eval_episodes)):
        with torch.no_grad():
            ret, length, _, _ = evaluate_episode(
                test_env,
                env_name,
                state_dim,
                act_dim,
                action_type,
                model,
                model_type,
                max_ep_len,
                scale,
                state_mean,
                state_std,
                target_return / scale,
                adv_act_dim=adv_act_dim,
                normalize_states=normalize_states,
                worst_case=True,
                device=device
            )
        returns.append(ret)
        lengths.append(length)

    return returns, lengths
