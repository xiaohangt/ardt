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
from decision_transformer.decision_transformer.training.trainer import TrainConfigs
from decision_transformer.decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.decision_transformer.training.seq_trainer import SequenceTrainer
from decision_transformer.decision_transformer.training.adv_seq_trainer import AdvSequenceTrainer
from decision_transformer.decision_transformer.utils.preemption import PreemptionManager


def eval_episodes(target_return):
        is_argmax = variant["argmax"]
        num_eval_episodes = 1 if variant['argmax'] else variant['num_eval_episodes']
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
                                    target_return=target_return/scale,
                                    mode=variant.get('mode', 'normal'),
                                    state_mean=state_mean,
                                    state_std=state_std,
                                    device=device,
                                )
                    returns.append(ret)
                    lengths.append(length)
            else:
                returns, lengths = evaluate(
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
                    variant['normalize_states'], 
                    batch_size=None
                )
            
            show_res_dict = {
                f'target_{target_return}_return_mean': np.mean(returns),
                f'target_{target_return}_return_std': np.std(returns),
            }

            result_dict = {
                f'target_{target_return}_return_mean': np.mean(returns),
                f'target_{target_return}_return_std': np.std(returns),
                f'target_{target_return}_length_mean': np.mean(lengths),
                f'target_{target_return}_length_std': np.std(lengths),
            }

            # dump information to that file
            rtg_path = variant['ret_file'][variant['ret_file'].rfind('/') + 1:]
            d_name = variant["data_name"][variant["data_name"].rfind('/') + 1:] if '/' in variant["data_name"] else variant["data_name"]

            test_adv = variant["test_adv"][variant["test_adv"].rfind('/') + 1:] if '/' in variant["test_adv"] else variant["test_adv"]
            traj_len = variant["traj_len"]
            seed = variant['seed']
            env_alpha = env.env_alpha if hasattr(env, 'env_alpha') else None
            
            if variant['algo'] != 'dt':
                save_path = f'results/{rtg_path}_traj{traj_len}_model{model_type}_adv{test_adv}_alpha{env_alpha}_{is_argmax}_{target_return}_{seed}.pkl'
            else:
                added_data_name = variant['added_data_name']
                added_data_prop = variant['added_data_prop']
                algo = variant['algo']
                save_path = f'results/{algo}_original_{d_name}_{added_data_name}_{added_data_prop}_traj{traj_len}_model{model_type}_adv{test_adv}_alpha{env_alpha}_{is_argmax}_{target_return}_{seed}.pkl'
                
            pickle.dump(result_dict, open(save_path, 'wb'))
            print("Evaluation results", show_res_dict, "saved to ", save_path)
            return show_res_dict
        return fn


def _discount_cumsum(x, gamma):
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
    # admin
    pm = PreemptionManager(variant['checkpoint_dir'], checkpoint_every=600)

    device = variant.get('device', 'cpu')
    log_to_wandb = variant.get('log_to_wandb', False)

    env_name = variant['env_name']
    model_type = variant['model_type']

    if model_type == 'bc':
        # since BC ignores target, no need for different evaluations
        env_targets = env_targets[:1]
    
    rtg_path = variant['ret_file'][variant['ret_file'].rfind('/') + 1:]
    
    # dimensionality
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

    print('=' * 50)
    print(f'Starting new experiment: {env_name}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)
    pickle.dump([np.mean(returns), np.std(returns)], open(f'offline_data/data_profile_{rtg_path}.pkl', 'wb'))
    print(f"offline_data/data_profile_{rtg_path}.pkl")

    # set up training
    # save all path information into separate lists
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        # pre-compute the return-to-gos
        path['rtg'] = _discount_cumsum(path['rewards'], gamma=1.)
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
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    num_timesteps = sum(traj_lens)

    # only train on top top_pct_traj trajectories (for %BC experiment)
    top_pct_traj = variant.get('top_pct_traj', 1.)
    num_timesteps = max(int(top_pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)
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

    # now gather train configs
    train_configs = TrainConfigs(
        action_dim=act_dim,
        adv_action_dim=adv_act_dim,
        action_type=action_type,
        state_dim=state_dim,
        state_mean=state_mean,
        state_std=state_std,
        returns_scale=scale,
        top_pct_traj=top_pct_traj,
        episode_length=max_ep_len,
        normalize_states=variant['normalize_states'],
    )

    # set up model
    if model_type == 'dt':
        model = pm.load_torch(
            'model', 
            DecisionTransformer,
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=variant['K'],
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
            max_length=variant['K'],
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
            max_length=variant['K'],
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer']
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

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
        lambda steps: min((steps + 1) / variant['warmup_steps'], 1)
    )

    # build trainer
    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            gradients_clipper=(lambda x: torch.nn.utils.clip_grad_norm_(x, variant['grad_clip_norm']))
            context_size=variant['K'],
            with_adv_action=False,
            env_name=env_name,
            trajectories=trajectories,
            trajectories_sorted_idx=sorted_inds,
            trajectories_sorted_probs=p_sample,
            train_configs=train_configs,
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    elif model_type == 'adt':
        trainer = AdvSequenceTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            gradients_clipper=(lambda x: torch.nn.utils.clip_grad_norm_(x, variant['grad_clip_norm'])),
            context_size=variant['K'],
            with_adv_action=True,
            env_name=env_name,
            trajectories=trajectories,
            train_configs=train_configs,
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    elif model_type == 'bc':
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            gradients_clipper=None,
            context_size=variant['K'],
            with_adv_action=False,
            env_name=env_name,
            trajectories=trajectories,
            train_configs=train_configs,
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    # Load learned returns-to-go
    if variant['algo'] in ['radt', 'esper']:
        assert variant['ret_file']
        with open(variant['ret_file'], 'rb') as f:
            rtg_dict = pickle.load(f)
        for i, path in enumerate(trajectories):
            path['rtg'] = rtg_dict[i]

    # Train
    completed_iters = pm.load_if_exists('completed_iters', 0)
    print("Trained for iterations:", completed_iters)
    for iter in range(completed_iters, variant['train_iters']):
        outputs = trainer.train_iteration(
            num_steps=variant['num_steps_per_iter'], 
            iter_num=iter+1,
            device=device,
            print_logs=True
        )
        pm.save('optimizer', optimizer.state_dict())
        pm.save('scheduler', scheduler.state_dict())
        pm.save('model', model.state_dict())
        pm.checkpoint()
        if log_to_wandb:
            wandb.log(outputs)
        completed_iters += 1
        pm.save('completed_iters', completed_iters)
        
    # Evaluate
    if not variant['is_training_only']:
        for tar in env_targets:
            eval_func = eval_episodes(tar)
            print(tar, eval_func(model))
