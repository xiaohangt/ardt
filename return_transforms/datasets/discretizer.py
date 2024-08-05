from copy import deepcopy

import numpy as np


class TrajectoryDiscretizer:
    def __init__(self, data, N_obs, N_acts):
        self.data = data
        self.N_obs = N_obs
        self.N_acts = N_acts

        all_states, all_actions, all_adv_actions = [], [], []
        for traj in data:
            all_states = all_states + traj.obs
            all_actions = all_actions + traj.actions
            all_adv_actions = all_adv_actions + [info_dict['adv'] for info_dict in traj.infos]
        all_states, all_actions, all_adv_actions = np.array(all_states), np.array(all_actions), np.array(all_adv_actions)
        
        self.obs_dim, self.acts_dim, self.adv_acts_dim = len(all_states[0]), len(all_actions[0]), len(all_adv_actions[0])

        obs_n_points_per_bin = int(np.ceil(len(all_states) / N_obs))
        acts_n_points_per_bin = int(np.ceil(len(all_states) / N_acts))
        adv_acts_n_points_per_bin = int(np.ceil(len(all_states) / N_acts))

        self.obs_thresholds = np.zeros([N_obs + 1, self.obs_dim])
        self.acts_thresholds = np.zeros([N_acts + 1, self.acts_dim])
        self.adv_acts_thresholds = np.zeros([N_acts + 1, self.adv_acts_dim])

        # observations
        for dimension in range(self.obs_dim):
            obs_sorted = np.sort(all_states[:, dimension])
            obs_thresholds = obs_sorted[::obs_n_points_per_bin]
            self.obs_thresholds[:, dimension] = np.concatenate([obs_thresholds, obs_sorted[-1:] + 0.01], axis=0)

        # actions
        for dimension in range(self.acts_dim):
            acts_sorted = np.sort(all_actions[:, dimension])
            acts_thresholds = acts_sorted[::acts_n_points_per_bin]
            self.acts_thresholds[:, dimension] = np.concatenate([acts_thresholds, acts_thresholds[-1:] + 0.01], axis=0)

        # adv actions
        for dimension in range(self.adv_acts_dim):
            adv_acts_sorted = np.sort(all_adv_actions[:, dimension])
            adv_acts_thresholds = acts_sorted[::adv_acts_n_points_per_bin]
            self.adv_acts_thresholds[:, dimension] = np.concatenate([adv_acts_thresholds, adv_acts_sorted[-1:] + 0.01], axis=0)

        # Discretizing:
        discrete_traj = deepcopy(data)
        for traj in discrete_traj:
            # obs
            for i, observation in enumerate(traj.obs):
                inds = np.argmax(~(observation >= self.obs_thresholds), axis=0) - 1
                one_hot_inds = np.eye(N_obs)[inds]
                discrete_obs = np.concatenate(one_hot_inds, axis=0).tolist()
                traj.obs[i] = discrete_obs

            # acts
            for i, action in enumerate(traj.actions):
                inds = np.argmax(~(action >= self.acts_thresholds), axis=0) - 1
                one_hot_inds = np.eye(N_acts)[inds]
                discrete_acts = np.concatenate(one_hot_inds, axis=0).tolist()
                traj.actions[i] = discrete_acts

            # adv acts
            for i, info in enumerate(traj.infos):
                inds = np.argmax(~(info['adv'] >= self.adv_acts_thresholds), axis=0) - 1
                one_hot_inds = np.eye(N_acts)[inds]
                discrete_adv_acts = np.concatenate(one_hot_inds, axis=0).tolist()
                traj.infos[i] = {'adv': discrete_adv_acts}

        self.discrete_traj = discrete_traj
        self.discrete_obs_dim = len(discrete_traj[0].obs[0])
        self.discrete_acts_dim = len(discrete_traj[0].actions[0])
        self.discrete_adv_acts_dim = len(discrete_traj[0].infos[0]['adv'])

    def reconstruct(self):
        pass