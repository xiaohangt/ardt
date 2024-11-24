import gym
import pickle

import numpy as np
import torch
import wandb

from decision_transformer.decision_transformer.evaluation.eval_fn_generator import EvalFnGenerator
from decision_transformer.decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.decision_transformer.models.adversarial_decision_transformer import AdversarialDecisionTransformer
from decision_transformer.decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.decision_transformer.training.trainer import TrainConfigs
from decision_transformer.decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.decision_transformer.training.seq_trainer import SequenceTrainer
from decision_transformer.decision_transformer.training.adv_seq_trainer import AdvSequenceTrainer
from decision_transformer.decision_transformer.utils.preemption import PreemptionManager
from stochastic_offline_envs.stochastic_offline_envs.envs.offline_envs.base import BaseOfflineEnv


def experiment(
        task: BaseOfflineEnv,
        env: gym.Env,
        max_ep_len: int,
        env_targets: list,
        scale: float,
        action_type: str,
        variant: dict,
    ):
    """
    Run the experiment for the given task and environment.

    Args:
        task (BaseOfflineEnv): Task object.
        env (gym.Env): Environment object.
        max_ep_len (int): Maximum episode length.
        env_targets (list): List of target environments.
        scale (float): Scaling factor for the states.
        action_type (str): Type of action space.
        variant (dict): Dictionary containing the experiment configuration.
    """

    def _discount_cumsum(x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
        return discount_cumsum

    # Admin
    pm = PreemptionManager(variant['checkpoint_dir'], checkpoint_every=600)
    device = variant['device']

    # Environment and model types
    env_name = variant['env_name']
    model_type = variant['model_type']
    if model_type == 'bc':
        # since BC ignores target, no need for different evaluations
        env_targets = env_targets[:1]
        
    # Dimensionality
    state_dim = int(np.prod(env.observation_space.shape))
    if action_type == 'discrete':
        act_dim = env.action_space.n
        adv_act_dim = env.adv_action_space.n
    else:
        act_dim = env.action_space.shape[0]
        adv_act_dim = env.adv_action_space.shape[0]

    # Load dataset
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
            adv_a = np.array([info["adv"] if info else -1 for info in traj.infos])
        elif "adv_action" in traj.infos[0]:
            adv_a = np.array([info["adv_action"] if info else 0 for info in traj.infos])
        else:
            print("WARNING: No adv or adv_action in infos")
            if action_type == "discrete":
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

    # Save all path information into separate lists and pre-compute returns
    states, traj_lens, returns = [], [], []

    for path in trajectories:
        path['rtg'] = _discount_cumsum(path['rewards'], gamma=1.)
        if env_name == "connect_four":
            cur_state = np.array([obs for obs in path['observations']])
            states.append(cur_state.reshape(cur_state.shape[0], -1))
        else:
            states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())

    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # Used for input normalisation
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    num_timesteps = sum(traj_lens)

    # Only train on top top_pct_traj trajectories (for %BC experiment)
    top_pct_traj = variant.get('top_pct_traj', 1.)
    num_timesteps = max(int(top_pct_traj * num_timesteps), 1)
    sorted_idx = np.argsort(returns)
    num_trajectories = 1
    timesteps = traj_lens[sorted_idx[-1]]
    idx = len(trajectories) - 2

    while idx >= 0 and timesteps + traj_lens[sorted_idx[idx]] <= num_timesteps:
        timesteps += traj_lens[sorted_idx[idx]]
        num_trajectories += 1
        idx -= 1

    sorted_idx = sorted_idx[-num_trajectories:]
    p_sample = traj_lens[sorted_idx] / sum(traj_lens[sorted_idx])

    # Some logging
    print('=' * 50)
    print(f'Starting new experiment: {env_name}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')

    print('=' * 50)
    returns_filename = variant['ret_file'][variant['ret_file'].rfind('/') + 1:]
    pickle.dump([np.mean(returns), np.std(returns)], open(f'offline_data/data_profile_{returns_filename}.pkl', 'wb'))
    print(f"offline_data/data_profile_{returns_filename}.pkl")

    # Used to generate evaluation functions to be fed into different training runs
    eval_fn_generator = EvalFnGenerator(
        variant.get('seed', 0),
        env_name,
        task,
        (1 if variant['argmax'] else variant['num_eval_episodes']),
        state_dim, 
        act_dim, 
        adv_act_dim,
        action_type,
        max_ep_len,
        scale, 
        state_mean, 
        state_std,
        variant['batch_size'], 
        variant['normalize_states'],
        device,
        variant['algo'],
        variant['ret_file'],
        variant['data_name'],
        variant['test_adv'],
        variant['added_data_name'],
        variant['added_data_prop']
    )

    # Now gather train configs
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
        batch_size=variant['batch_size'],
        normalize_states=variant['normalize_states'],
    )

    # Set up model
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

    # Build trainer
    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            model_type=model_type,
            optimizer=optimizer,
            scheduler=scheduler,
            gradients_clipper=(lambda x: torch.nn.utils.clip_grad_norm_(x, variant['grad_clip_norm'])),
            context_size=variant['K'],
            with_adv_action=False,
            env_name=env_name,
            num_trajectories=num_trajectories,
            trajectories=trajectories,
            trajectories_sorted_idx=sorted_idx,
            trajectories_sorted_probs=p_sample,
            train_configs=train_configs,
            eval_fns=[eval_fn_generator.generate_eval_fn(tgt) for tgt in env_targets],
        )
    elif model_type == 'adt':
        trainer = AdvSequenceTrainer(
            model=model,
            model_type=model_type,
            optimizer=optimizer,
            scheduler=scheduler,
            gradients_clipper=(lambda x: torch.nn.utils.clip_grad_norm_(x, variant['grad_clip_norm'])),
            context_size=variant['K'],
            with_adv_action=True,
            env_name=env_name,
            num_trajectories=num_trajectories,
            trajectories=trajectories,
            trajectories_sorted_idx=sorted_idx,
            trajectories_sorted_probs=p_sample,
            train_configs=train_configs,
            eval_fns=[eval_fn_generator.generate_eval_fn(tgt) for tgt in env_targets],
        )
    elif model_type == 'bc':
        trainer = ActTrainer(
            model=model,
            model_type=model_type,
            optimizer=optimizer,
            scheduler=scheduler,
            gradients_clipper=None,
            context_size=variant['K'],
            with_adv_action=False,
            env_name=env_name,
            num_trajectories=num_trajectories,
            trajectories=trajectories,
            trajectories_sorted_idx=sorted_idx,
            trajectories_sorted_probs=p_sample,
            train_configs=train_configs,
            eval_fns=[eval_fn_generator.generate_eval_fn(tgt) for tgt in env_targets],
        )

    # Load learned returns-to-go
    if variant['algo'] in ['ardt', 'esper']:
        assert variant['ret_file']
        with open(f"{variant['ret_file']}.pkl", 'rb') as f:
            rtg_dict = pickle.load(f)
        for i, path in enumerate(trajectories):
            path['rtg'] = rtg_dict[i]

    # Train the model on learned targets
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
        if variant.get('log_to_wandb', False):
            wandb.log(outputs)
        completed_iters += 1
        pm.save('completed_iters', completed_iters)
        
    # Evaluate trained model
    if not variant['is_training_only']:
        for tgt in env_targets:
            eval_fn = eval_fn_generator.generate_eval_fn(tgt)
            print(tgt, eval_fn(model=model, model_type=model_type))
