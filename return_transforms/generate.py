import pickle
from pathlib import Path

import gym
import numpy as np
import yaml

from data_loading.load_mujoco import Trajectory
from return_transforms.algos.esper.esper import esper
from return_transforms.algos.maxmin.maxmin import maxmin


def _normalize_obs(trajs: list[Trajectory]) -> list[Trajectory]:
    """
    Normalize the observations in the given list of trajectories.

    Args:
        trajs (list[Trajectory]): List of trajectory objects to normalize.

    Returns:
        list[Trajectory]: The same list of trajectories, but with normalized observations.
    """
    # Collect all observations from all trajectories
    obs_list = []
    for traj in trajs:
        obs_list.extend(traj.obs)
    
    # Compute mean and standard deviation of observations
    obs = np.array(obs_list)
    obs_mean = np.mean(obs, axis=0)
    obs_std = np.std(obs, axis=0) + 1e-8 

    # Normalize each observation in each trajectory
    for traj in trajs:
        for i in range(len(traj.obs)):
            traj.obs[i] = (traj.obs[i] - obs_mean) / obs_std

    # Return normalized trajectories
    return trajs


def generate_expected(
        env: gym.Env, 
        trajs: list[Trajectory], 
        config: dict, 
        ret_file: str, 
        device: str, 
        n_cpu: int
    ):
    """
    Generate expected returns using the ESPER algorithm and save them to a file.

    Args:
        env (gym.Env): The environment used for evaluation.
        trajs (list[Trajectory]): List of trajectories to process.
        config (dict): Configuration dictionary for ESPER.
        ret_file (str): Path to save the generated returns.
        device (str): The device to run the computations on ('cpu' or 'cuda').
        n_cpu (int): Number of CPUs to use for parallel processing.
    """
    # Load the configuration from the YAML file
    config = yaml.safe_load(Path(config).read_text())
    
    # Ensure the selected method is 'esper'
    assert config['method'] == 'esper', "ESPER is the algo to use to learn expected returns."

    # Normalize observations if specified in the config
    if config['normalize']:
        trajs = _normalize_obs(trajs)
    
    # Run the ESPER algorithm to compute the returns
    print('Generating ESPER returns...')

    rets = esper(
        trajs,
        env.action_space,
        config['dynamics_model_args'],
        config['cluster_model_args'], 
        config['train_args'], 
        device,
        n_cpu
    )

    # Save the generated returns to the specified file
    print(f'Done. Saving returns to {ret_file}.')
    Path(ret_file).parent.mkdir(parents=True, exist_ok=True)  
    with open(f"{ret_file}.pkl", 'wb') as f:
        pickle.dump(rets, f) 


def generate_maxmin(
        env: gym.Env, 
        trajs: list[Trajectory], 
        config: dict, 
        ret_file: str, 
        device: str, 
        n_cpu: int,
        is_simple_model: bool = False, 
        is_toy: bool = False
    ):
    """
    Generate worst-case returns using the ARDT algorithm and save them to a file.

    Args:
        env (gym.Env): The environment used for evaluation.
        trajs (list[Trajectory]): List of trajectories to process.
        config (dict): Configuration dictionary for ARDT.
        ret_file (str): Path to save the generated returns.
        device (str): The device to run the computations on ('cpu' or 'cuda').
        n_cpu (int): Number of CPUs to use for parallel processing.
        is_simple_model (bool, optional): Whether to use a simple model for ARDT. Default is False.
        is_toy (bool, optional): Whether the environment is a toy environment. Default is False.
    """
    # Load the configuration from the YAML file
    config = yaml.safe_load(Path(config).read_text())
    
    # Ensure the selected method is 'ardt'
    assert config['method'] == 'ardt', "ARDT is the algo to use to learn worst-case returns."

    # Normalize observations if specified in the config
    if config['normalize']:
        trajs = _normalize_obs(trajs)
    
    # Run the ARDT (maxmin) algorithm to compute worst-case returns and prompt values
    print('Generating ARDT returns...')
    
    rets, prompt_value = maxmin(
        trajs,
        env.action_space,
        env.adv_action_space,  
        config['train_args'], 
        device,
        n_cpu,
        is_simple_model=is_simple_model,  
        is_toy=is_toy  
    )

    # Save the generated returns and prompt values to the specified file
    print(f'Done. Saving returns and prompts to {ret_file}.')
    Path(ret_file).parent.mkdir(parents=True, exist_ok=True) 
    
    # Save returns
    with open(f"{ret_file}.pkl", 'wb') as f:
        pickle.dump(rets, f)
    
    # Save prompt values
    with open(f"{ret_file}_prompt.pkl", 'wb') as f:
        pickle.dump(prompt_value, f)
    