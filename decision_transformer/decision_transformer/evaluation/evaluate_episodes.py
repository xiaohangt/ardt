import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

from stochastic_offline_envs.envs.connect_four.connect_four_env import GridWrapper


def worst_case_env_step(state, action, t, env):
    class_name = env.__class__.__name__
    new_state_ind = -1
    adv_action = np.random.choice(2, 1)
    _, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    
    if class_name == "GamblingEnv":
        if t == 0:
            if action == 0:
                w_reward = -15 
            elif action == 1:
                w_reward = -6
            else:
                w_reward = 1
        elif t == 1:
            reward = w_reward
            assert done
    elif class_name == "ToyEnv":
        if t == 0:
            if action == 0:
                env.w_reward = 0
            elif action == 1:
                env.w_reward = 1
            else:
                raise Exception("Action Error")
        else:
            reward = env.w_reward
            assert done
    elif class_name == "MSToyEnv":
        done = False
        if t == 0:
            if action > 0:
                reward = 4
                done = True
            else:
                new_state_ind = 1
                reward = 0
                adv_action = 0
        else:
            try:
                reward = env.reward_list[action + (state.argmax() - 1) * 3]
            except:
                breakpoint()
            done = True
    elif class_name == "NewMSToyEnv":
        done = False
        if t == 0:
            if action > 0:
                reward = 4
                done = True
            else:
                new_state_ind = 1
                reward = 0
        else:
            try:
                reward = env.reward_list[action * 2 + (state.argmax() - 1) * 3]
            except:
                breakpoint()
            done = True
    else:
        raise Exception("Env Error")

    new_state = np.eye(state.size)[new_state_ind] if new_state_ind != -1 else state
    return new_state, reward, done, False, {"adv_action": adv_action}


def evaluate(
        target_return, 
        model_type, 
        num_eval_episodes, 
        task, 
        state_dim, 
        act_dim, 
        adv_act_dim, 
        model, 
        max_ep_len, 
        scale, 
        state_mean, 
        state_std, 
        device, 
        action_type, 
        is_argmax, 
        normalize_states, 
        batch_size=None, 
        qsa2_model=None
    ):
    mode = 'normal'
    if batch_size:
        if test_env.__class__.__name__ == "ConnectFourEnv":
            envs = [GridWrapper(task.test_env_cls()) for _ in range(batch_size)]
        else:
            envs = [task.test_env_cls() for _ in range(batch_size)]
        with torch.no_grad():
            returns, lengths = evaluate_episode_batch(
                batch_size, 
                envs, 
                state_dim, 
                act_dim, 
                model, 
                max_ep_len, 
                scale, 
                state_mean, 
                state_std, 
                device, 
                target_return, 
                mode, 
                action_type, 
                model_type
            )
        return returns, lengths
    
    returns, lengths = [], []
    test_env = task.test_env_cls()
    if test_env.__class__.__name__ == "ConnectFourEnv":
        test_env = GridWrapper(test_env) 

    for _ in tqdm(range(num_eval_episodes)):
        with torch.no_grad():
            ret, length, action_probs, states = evaluate_episode_rtg(
                test_env,
                state_dim,
                act_dim,
                model,
                max_ep_len=max_ep_len,
                scale=scale,
                target_return=target_return / scale,
                state_mean=state_mean,
                state_std=state_std,
                device=device,
                action_type=action_type,
                worst_case=True,
                argmax=is_argmax,
                normalize_states=normalize_states,
                model_type=model_type,
                adv_act_dim=adv_act_dim,
                qsa2_model=qsa2_model
            )
        returns.append(ret)
        lengths.append(length)

    return returns, lengths


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        action_type='continuous',
        worst_case=True,
        argmax=False,
        adv_act_dim=None,
        model_type='dt',
        normalize_states=False,
        qsa2_model=None
    ):
    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state, _ = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    if not adv_act_dim:
        adv_act_dim = act_dim
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    adv_actions = torch.zeros((0, adv_act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    # ep_return = target_return
    # target_return = torch.zeros((1, 0), device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

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
                normalized_states.to(dtype=torch.float32), 
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
                batch_size=1
            )
        elif model_type == 'adt':
            action = model.get_action(
                normalized_states.to(dtype=torch.float32),
                actions.to(dtype=torch.float32),
                adv_actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
                batch_size=1
            )
        action = action[0, -1]

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

        if worst_case and (env.__class__.__name__ in ["GamblingEnv", "ToyEnv", "MSToyEnv", "NewMSToyEnv"]):
            state, reward, terminated, truncated, infos = worst_case_env_step(state, action, t, env)
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

        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length, None, None


def evaluate_episode_batch(
        batch_size,
        envs,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        action_type='continuous',
        model_type='dt'
    ):
    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)
    states = np.array([env.reset()[0] for env in envs])

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(states).reshape(batch_size, 1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((batch_size, 1, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros((batch_size, 1, 1), device=device, dtype=torch.float32)
    target_return = torch.full((batch_size, 1, 1), target_return, device=device, dtype=torch.float32)
    timesteps = torch.zeros((batch_size, 1, 1), device=device, dtype=torch.long)

    episode_return, episode_length, dones_list = np.zeros(batch_size), np.zeros(batch_size), np.zeros(batch_size)
    for t in tqdm(range(max_ep_len)):
        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            batch_size=batch_size
        )
        action = action[:,-1]

        if action_type == 'discrete':
            # sample action
            act_probs = F.softmax(action, dim=-1)
            action = torch.multinomial(act_probs, 1, replacement=True).squeeze(-1)

        action = action.detach().cpu().numpy()
        s_list, r_list = [], []
        for i, env in enumerate(envs):
            if dones_list[i] == 0:
                s, r, terminated, truncated, _ = env.step(action[i])
                done = terminated or truncated
                s_list.append(s)
                r_list.append(r)
                episode_return[i] += r
                episode_length[i] += 1
                dones_list[i] = 1 if done else 0
            else:
                s_list.append(np.zeros_like(s))
                r_list.append(0)
                
        cur_reward = torch.from_numpy(np.array(r_list)).to(device=device).reshape(-1, 1, 1)
        cur_state = torch.from_numpy(np.array(s_list)).to(device=device).reshape(-1, 1, state_dim)
        states = torch.cat([states, cur_state], dim=1)
        rewards = torch.cat([rewards, cur_reward], dim=1)

        pred_return = target_return[:,-1] - (cur_reward[:, 0] / scale)
            
        target_return = torch.cat([target_return, pred_return.reshape(batch_size, 1, 1)], dim=1)
        timesteps = torch.cat([timesteps, torch.ones((batch_size, 1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        if np.all(dones_list == 1):
            break

    return episode_return, episode_length


def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
        action_type="discrete"
    ):
    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)
    state, _ = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)

    episode_return, episode_length = 0, 0
    for _ in range(max_ep_len):
        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )

        if action_type == 'discrete':
            action = Categorical(probs=F.softmax(action, dim=0)).sample()
            one_hot_action = torch.zeros(1, act_dim).float()
            one_hot_action[0, action] = 1
            actions[-1] = one_hot_action
        else:
            actions[-1] = action

        action = action.detach().cpu().numpy()
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length