import argparse


def load_env(env_name, data_name=None):
    if env_name == 'connect_four':
        from stochastic_offline_envs.envs.offline_envs.connect_four_offline_env import ConnectFourOfflineEnv
        from stochastic_offline_envs.envs.connect_four.connect_four_env import GridWrapper
        task = ConnectFourOfflineEnv(data_name=data_name)
        env = task.env_cls()
        env = GridWrapper(env)
        trajs = task.trajs
        for traj in trajs:
            for i in range(len(traj.obs)):
                traj.obs[i] = traj.obs[i]['grid']
        return env, trajs
    elif env_name == 'tfe':
        from stochastic_offline_envs.envs.offline_envs.tfe_offline_env import TFEOfflineEnv
        task = TFEOfflineEnv()
        env = task.env_cls()
        trajs = task.trajs
        return env, trajs
    elif env_name == 'gambling':
        from stochastic_offline_envs.envs.offline_envs.gambling_offline_env import GamblingOfflineEnv
        task = GamblingOfflineEnv()
        env = task.env_cls()
        trajs = task.trajs
        return env, trajs
    elif env_name == 'toy':
        from stochastic_offline_envs.envs.offline_envs.toy_offline_env import ToyOfflineEnv
        task = ToyOfflineEnv()
        env = task.env_cls()
        trajs = task.trajs
        return env, trajs
    elif env_name == "mstoy":
        from stochastic_offline_envs.envs.offline_envs.mstoy_offline_env import MSToyOfflineEnv
        task = MSToyOfflineEnv()
        env = task.env_cls()
        trajs = task.trajs
        return env, trajs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default=None)
    parser.add_argument('--env_name', type=str, default='connect_four')
    parser.add_argument('--config', type=str, default='"configs/esper/connect_four.yaml"')
    parser.add_argument('--ret_file', type=str, default='test')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--n_cpu', type=int, default=10)

    args = parser.parse_args()

    data_name, env_name, config, ret_file, device, n_cpu = args.data_name, args.env_name, args.config, args.ret_file, args.device, args.n_cpu
    print(f"Loading offline RL task: {env_name}")
    load_env(env_name, data_name)
