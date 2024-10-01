import pickle

import numpy as np
import torch

from decision_transformer.decision_transformer.evaluation.evaluate_episodes import evaluate
from stochastic_offline_envs.stochastic_offline_envs.envs.offline_envs.base import BaseOfflineEnv


class EvalFnGenerator:
    """
    Class to generate evaluation functions for a given model and model type.

    Args:
        seed (int): Seed for reproducibility.
        env_name (str): Name of the environment.
        task (BaseOfflineEnv): Task object.
        num_eval_episodes (int): Number of episodes to evaluate.
        state_dim (int): Dimension of the state space.
        act_dim (int): Dimension of the action space.
        adv_act_dim (int): Dimension of the adversary action space.
        action_type (str): Type of action space.
        traj_len (int): Length of the trajectory.
        scale (bool): Whether to scale the states.
        state_mean (np.ndarray): Mean of the states.
        state_std (np.ndarray): Standard deviation of the states.
        batch_size (int): Batch size for evaluation.
        normalize_states (bool): Whether to normalize the states.
        device (torch.device): Device to run the evaluation on.
        algo_name (str): Name of the algorithm.
        returns_filename (str): Name of the returns file.
        dataset_name (str): Name of the dataset.
        test_adv_name (str): Name of the adversary.
        added_dataset_name (str): Name of the added dataset.
        added_dataset_prop (float): Proportion of the added dataset.
    """
    def __init__(
            self,
            seed: int,
            env_name: str,
            task: BaseOfflineEnv,
            num_eval_episodes: int,
            state_dim: int,
            act_dim: int,
            adv_act_dim: int,
            action_type: str,
            traj_len: int,
            scale: float,
            state_mean: float,
            state_std: float,
            batch_size: int,
            normalize_states: bool,
            device: torch.device,
            algo_name: str,
            returns_filename: str,
            dataset_name: str,
            test_adv_name: str,
            added_dataset_name: str,
            added_dataset_prop: float
        ):
        self.seed = seed
        self.env_name = env_name
        self.task = task
        self.num_eval_episodes = num_eval_episodes
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.adv_act_dim = adv_act_dim
        self.action_type = action_type
        self.traj_len = traj_len
        self.scale = scale
        self.state_mean = state_mean
        self.state_std = state_std
        self.target_return = None
        self.batch_size = batch_size
        self.normalize_states = normalize_states
        self.device = device
        self.storage_path = self._build_storage_path(
            algo_name,
            returns_filename,
            dataset_name,
            test_adv_name,
            added_dataset_name,
            added_dataset_prop
        )

    def _build_storage_path(
            self,
            algo_name: str,
            returns_filename: str,
            dataset_name: str,
            test_adv_name: str,
            added_dataset_name: str,
            added_dataset_prop: float
        ) -> str:
        env = self.task.test_env_cls()
        env_alpha = env.env_alpha if hasattr(env, 'env_alpha') else None

        test_adv_name = (
            test_adv_name[test_adv_name.rfind('/') + 1:] 
            if '/' in test_adv_name
            else test_adv_name
        )
        
        if algo_name != 'dt':
            returns_filename = returns_filename[returns_filename.rfind('/') + 1:]
            dataset_name = (
                dataset_name[dataset_name.rfind('/') + 1:] 
                if '/' in dataset_name
                else dataset_name
            )
            return (
                f'results/{returns_filename}_traj{self.traj_len}_model/model_type/_adv{test_adv_name}_' +
                f'alpha{env_alpha}_False_{self.target_return}_{self.seed}.pkl'
            )
        else:
            return (
                f'results/{algo_name}_original_{dataset_name}_{added_dataset_name}_' +
                f'{added_dataset_prop}_traj{self.traj_len}_model/model_type/_' + 
                f'adv{test_adv_name}_alpha{env_alpha}_False_{self.target_return}_{self.seed}.pkl'
            )

    def _eval_fn(self, model: torch.nn.Module, model_type: str) -> dict:
        returns, lengths = evaluate(
            self.env_name,
            self.task,
            self.num_eval_episodes,
            self.state_dim, 
            self.act_dim, 
            self.adv_act_dim,
            self.action_type,
            model, 
            model_type,
            self.traj_len,
            self.scale, 
            self.state_mean, 
            self.state_std,
            self.target_return,
            batch_size=self.batch_size, 
            normalize_states=self.normalize_states,
            device=self.device
        )
        
        show_res_dict = {
            f'target_{self.target_return}_return_mean': np.mean(returns),
            f'target_{self.target_return}_return_std': np.std(returns),
        }

        result_dict = {
            f'target_{self.target_return}_return_mean': np.mean(returns),
            f'target_{self.target_return}_return_std': np.std(returns),
            f'target_{self.target_return}_length_mean': np.mean(lengths),
            f'target_{self.target_return}_length_std': np.std(lengths),
        }

        run_storage_path = self.storage_path.replace("/model_type/", model_type)          
        pickle.dump(result_dict, open(run_storage_path, 'wb'))
        print("Evaluation results", show_res_dict, "saved to ", run_storage_path)
        return show_res_dict

    def generate_eval_fn(self, tgt_return: int) -> '_eval_fn':
        self.target_return = tgt_return
        return self._eval_fn
