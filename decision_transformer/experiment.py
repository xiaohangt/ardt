import pickle
import random

import gym
import numpy as np
import torch
import wandb
from tqdm import tqdm

from decision_transformer.decision_transformer.evaluation.evaluate_episodes import evaluate, evaluate_episode
from decision_transformer.decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.decision_transformer.models.adversarial_decision_transformer import AdversarialDecisionTransformer
from decision_transformer.decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.decision_transformer.training.seq_trainer import SequenceTrainer
from decision_transformer.decision_transformer.training.adv_seq_trainer import AdvSequenceTrainer
from decision_transformer.decision_transformer.utils.preemption import PreemptionManager


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def experiment(
        task,
        env,
        max_ep_len,
        env_targets,
        scale,
        action_type,
        variant
    ):
    pm = PreemptionManager(variant['checkpoint_dir'], checkpoint_every=600)

    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    env_name = variant['env_name']
    model_type = variant['model_type']

    ret_postprocess_fn = lambda returns: returns

    if model_type == 'bc':
        # since BC ignores target, no need for different evaluations
        env_targets = env_targets[:1]
    
    rtg_path = variant['ret_file'][variant['ret_file'].rfind('/') + 1:]
    state_dim = np.prod(env.observation_space.shape)
    if action_type == 'discrete':
        act_dim = env.action_space.n
        adv_act_dim = env.adv_action_space.n
    else:
        act_dim = env.action_space.shape[0]
        adv_act_dim = env.adv_action_space.shape[0]

    # load dataset
    raw_trajectories = task.trajs
    trajectories = []

    for i, traj in enumerate(raw_trajectories):
        traj_dict = {}
        for i, key in enumerate(["observations", "actions", "rewards"]):
            cur_info = traj[i]
            if key == "actions":
                traj_dict["dones"] = np.zeros(len(cur_info), dtype=bool)
                traj_dict["dones"][-1] = True
                traj_dict[key] = np.array(cur_info) if action_type != 'discrete' else np.eye(act_dim)[cur_info]
            else:
                traj_dict[key] = np.array(cur_info)

        if "adv" in traj.infos[0]:
            adv_a = np.array([info["adv"] if info else 0 for info in traj.infos])
        elif "adv_action" in traj.infos[0]:
            adv_a = np.array([info["adv_action"] if info else 0 for info in traj.infos])
        elif action_type == "discrete":
            adv_a = np.zeros((len(traj_dict["actions"])))
        else:
            adv_a = np.zeros((len(traj_dict["actions"]), adv_act_dim))

        if action_type == "discrete":
            traj_dict['adv_actions'] = np.zeros((len(traj_dict["actions"]), adv_act_dim))
            traj_dict['adv_actions'][np.arange(len(traj_dict["actions"])), adv_a.astype(int)] = 1
            if traj.infos[-1] == {}:
                traj_dict['adv_actions'][-1] = 0
        else:
            traj_dict['adv_actions'] = adv_a
        
        trajectories.append(traj_dict)

    # save all path information into separate lists
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        # pre-compute the return-to-gos
        path['rtg'] = discount_cumsum(path['rewards'], gamma=1.)
        if env_name == "connect_four":
            cur_state = np.array([obs for obs in path['observations']])
            states.append(cur_state.reshape(cur_state.shape[0], -1))
        else:
            states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(
        states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    pickle.dump([np.mean(returns), np.std(returns)], open(f'data/data_profile_{rtg_path}.pkl', 'wb'))
    print(f"data/data_profile_{rtg_path}.pkl")

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = 1 if variant['argmax'] else variant['num_eval_episodes']
    top_pct_traj = variant.get('top_pct_traj', 1.)

    # only train on top top_pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(top_pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, adv_a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            if env_name not in ["halfcheetah", "hopper", "walker2d"]:
                si = 0
            else:
                si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            if env_name == "connect_four":
                cur_state = np.array([obs for obs in traj['observations']])
            else:
                cur_state = traj['observations']
            s.append(cur_state[si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            adv_a.append(traj['adv_actions'][si:si + max_len].reshape(1, -1, adv_act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
            rtg.append(traj['rtg'][si:si + max_len].reshape(1, -1, 1))

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            if variant['normalize_states']:
                s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            adv_a[-1] = np.concatenate([np.zeros((1, max_len - tlen, adv_act_dim)), adv_a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len -  tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)),   rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(
            dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(
            dtype=torch.float32, device=device)
        adv_a = torch.from_numpy(np.concatenate(adv_a, axis=0)).to(
            dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(
            dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(
            dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(
            dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
            dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        if model_type == 'adt':
            return s, a, adv_a, r, d, rtg, timesteps, mask
        else:
            return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        is_argmax = variant["argmax"]
        def fn(model):
            print("Start evaluating:")
            if model_type == 'bc':
                returns, lengths = [], []
                for _ in tqdm(range(num_eval_episodes)):
                    with torch.no_grad():
                        ret, length = evaluate_episode(
                                    env,
                                    state_dim,
                                    act_dim,
                                    model,
                                    max_ep_len=max_ep_len,
                                    target_return=target_rew / scale,
                                    mode=variant.get('mode', 'normal'),
                                    state_mean=state_mean,
                                    state_std=state_std,
                                    device=device,
                                )
                    returns.append(ret_postprocess_fn(ret))
                    lengths.append(length)
            else:
                returns, lengths = evaluate(
                    target_rew, 
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
                    variant['normalize_states'], 
                    batch_size=None
                )
            
            show_res_dict = {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
            }

            result_dict = {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }

            # dump information to that file
            rtg_path = variant['ret_file'][variant['ret_file'].rfind('/') + 1:]
            d_name = variant["data_name"][variant["data_name"].rfind('/') + 1:] if '/' in variant["data_name"] else variant["data_name"]

            test_adv = variant["test_adv"][variant["test_adv"].rfind('/') + 1:] if '/' in variant["test_adv"] else variant["test_adv"]
            traj_len = variant["traj_len"]
            seed = variant['seed']
            env_alpha = env.env_alpha if hasattr(env, 'env_alpha') else None
            
            if variant['algo'] != 'dt':
                save_path = f'results/{rtg_path}_traj{traj_len}_model{model_type}_adv{test_adv}_alpha{env_alpha}_{is_argmax}_{target_rew}_{seed}.pkl'
            else:
                added_data_name = variant['added_data_name']
                added_data_prop = variant['added_data_prop']
                algo = variant['algo']
                save_path = f'results/{algo}_original_{d_name}_{added_data_name}_{added_data_prop}_traj{traj_len}_model{model_type}_adv{test_adv}_alpha{env_alpha}_{is_argmax}_{target_rew}_{seed}.pkl'
                
            pickle.dump(result_dict, open(save_path, 'wb'))
            print("Evaluation results", show_res_dict, "saved to ", save_path)
            return show_res_dict
        return fn

    if model_type == 'dt':
        model = pm.load_torch(
            'model', 
            DecisionTransformer,
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4 * variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
            action_tanh=action_type == 'continuous',
            rtg_seq=variant['rtg_seq']
        )
    elif model_type == 'adt':
        model = pm.load_torch(
            'model', 
            AdversarialDecisionTransformer,
            state_dim=state_dim,
            act_dim=act_dim,
            adv_act_dim=adv_act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4 * variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
            action_tanh=action_type == 'continuous',
            rtg_seq=variant['rtg_seq']
        )
    elif model_type == 'bc':
        model = pm.load_torch(
            'model', 
            MLPBCModel,
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer']
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = pm.load_torch(
        'optimizer', 
        torch.optim.AdamW,
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay']
    )
    scheduler = pm.load_torch(
        'scheduler', 
        torch.optim.lr_scheduler.LambdaLR,
        optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    if action_type == 'continuous':
        action_loss = lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2)
    else:
        ce_loss = torch.nn.CrossEntropyLoss()
        action_loss = lambda s_hat, a_hat, r_hat, s, a, r: ce_loss(a_hat, torch.argmax(a, dim=-1))

    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=action_loss,
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    elif model_type == 'adt':
        trainer = AdvSequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=action_loss,
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    elif model_type == 'bc':
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=action_loss,
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    if variant['algo'] in ['radt', 'esper']:
        assert variant['ret_file']
        # Load custom return-to-go
        # Load the pickle
        with open(variant['ret_file'], 'rb') as f:
            rtg_dict = pickle.load(f)
        for i, path in enumerate(trajectories):
            try:
                path['rtg'] = rtg_dict[i]
            except:
                breakpoint()

    completed_iters = pm.load_if_exists('completed_iters', 0)
    print("Trained for iterations:", completed_iters)
    for iter in range(completed_iters, variant['train_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
        pm.save_torch('optimizer', optimizer)
        pm.save_torch('scheduler', scheduler)
        pm.save_torch('model', model)
        pm.checkpoint()
        if log_to_wandb:
            wandb.log(outputs)
        completed_iters += 1
        pm.save('completed_iters', completed_iters)
        
    if not variant['is_training_only']:
        for tar in env_targets:
            eval_func = eval_episodes(tar)
            print(tar, eval_func(model))
