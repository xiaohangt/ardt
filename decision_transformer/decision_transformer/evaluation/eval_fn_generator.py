import pickle

import numpy as np
import torch

from decision_transformer.decision_transformer.evaluation.evaluate_episodes import evaluate


class EvalFnGenerator:

    def __init__(
            self,
            seed,
            task,
            env_name,
            num_eval_episodes,
            state_dim, 
            act_dim, 
            adv_act_dim,
            action_type,
            traj_len,
            scale, 
            state_mean, 
            state_std,
            batch_size, 
            normalize_states,
            returns_filename,
            dataset_name,
            test_adv_name,
            added_dataset_name,
            added_dataset_prop
        ):
        self.seed = seed
        self.task = task
        self.env_name = env_name
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
        self.storage_path = self._build_storage_path(
            returns_filename,
            dataset_name,
            test_adv_name,
            added_dataset_name,
            added_dataset_prop
        )

    def _build_storage_path(
            self,
            returns_filename,
            dataset_name,
            test_adv_name,
            added_dataset_name,
            added_dataset_prop
        ):
        env = self.task.test_env_cls()
        env_alpha = env.env_alpha if hasattr(env, 'env_alpha') else None

        test_adv_name = (
            test_adv_name[test_adv_name.rfind('/') + 1:] 
            if '/' in test_adv_name
            else test_adv_name
        )
        
        if self.algo != 'dt':
            returns_filename = returns_filename[returns_filename.rfind('/') + 1:]
            dataset_name = (
                dataset_name[dataset_name.rfind('/') + 1:] 
                if '/' in dataset_name
                else dataset_name
            )
            self.storage_path = (
                f'results/{returns_filename}_traj{self.traj_len}_model/model_type/_adv{test_adv_name}_' +
                f'alpha{env_alpha}_False_{self.target_return}_{self.seed}.pkl'
            )
        else:
            self.storage_path = (
                f'results/{self.algo}_original_{dataset_name}_{added_dataset_name}_' +
                f'{added_dataset_prop}_traj{self.traj_len}_model/model_type/_' + 
                f'adv{test_adv_name}_alpha{env_alpha}_False_{self.target_return}_{self.seed}.pkl'
            )

    def _eval_fn(self, model: torch.nn.Module, model_type: str) -> dict:
        returns, lengths = evaluate(
            self.task,
            self.env_name,
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
            device=model.device
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

        # dump information to that file
        run_storage_path = self.storage_path.replace("/model_type/", model_type)          
        pickle.dump(result_dict, open(run_storage_path, 'wb'))
        print("Evaluation results", show_res_dict, "saved to ", run_storage_path)
        return show_res_dict

    def generate_eval_fn(self, tgt_return: int) -> '_eval_fn':
        self.target_return = tgt_return
        return self._eval_fn
