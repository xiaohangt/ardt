import torch

from return_transforms.models.basic.mlp import MLP


class CustomNetworkConfig():
    context_size = 3
    state_dim = 7
    pr_act_dim = 3
    adv_act_dim = 3


class PolicyNetwork(torch.nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """
    def __init__(
        self,
        feature_dim: int=1,
        hidden_dim: int=64,
        last_layer_dim_pi: int = 2,
    ):
        super().__init__()
        self.config = CustomNetworkConfig()
        feature_dim = self.config.state_dim

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi

        self.obs_embed = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim), torch.nn.ReLU(),
        )

        self.r_embed = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim), torch.nn.ReLU(),
        )
        # Policy network
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, last_layer_dim_pi)
        )


    def forward(self, features: torch.Tensor, conditioned_r):
        return self.forward_actor(features, conditioned_r)

    def forward_actor(self, features: torch.Tensor, r) -> torch.Tensor:
        features_emd = self.obs_embed(features)
        r_emd = self.r_embed(r)
        return torch.torch.nn.functional.softmax(self.policy_net(torch.cat([features_emd, r_emd], dim=-1)), dim=-1)


class AdvPolicyNetwork(torch.nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """
    def __init__(
        self,
        obs_dim: int=1,
        action_dim: int=3,
        adv_action_dim: int=7,
        hidden_dim: int=64,
        last_layer_dim_pi: int = 2,
    ):
        super().__init__()
        self.state_dim = obs_dim
        self.act_dim = action_dim
        self.adv_action_dim = adv_action_dim

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.obs_embed = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, hidden_dim), torch.nn.ReLU(),
        )

        self.act_embed = torch.nn.Sequential(
            torch.nn.Linear(self.act_dim, hidden_dim), torch.nn.ReLU(),
        )
        # Policy network
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, self.adv_action_dim)
        )


    def forward(self, features: torch.Tensor, pr_action):
        return self.forward_actor(features, pr_action)

    def forward_actor(self, features: torch.Tensor, pr_action) -> torch.Tensor:
        features_emd = self.obs_embed(features)
        pr_action_emd = self.act_embed(pr_action)
        return torch.torch.nn.functional.softmax(self.policy_net(torch.cat([features_emd, pr_action_emd], dim=-1)), dim=-1)


class NewRtgNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, adv_action_dim, model_args, hidden_dim=64, include_adv=False, is_lstm=True) -> None:
        super().__init__()
        self.include_adv = include_adv
        self.is_lstm = is_lstm

        hidden_dim = model_args['ret_obs_action_model']['hidden_size']
        self.hidden_dim = hidden_dim

        input_dim = state_dim + action_dim + adv_action_dim if include_adv else state_dim + action_dim
        self.ret_obs_action_model = MLP(input_dim, hidden_dim, **model_args['ret_obs_action_model'])
        
        if is_lstm:
            self.lstm_model = torch.nn.LSTM(hidden_dim, hidden_dim,batch_first=True)
            self.return_model = torch.nn.Linear(hidden_dim, 1)
        else:
            self.return_model = MLP(hidden_dim, 1, **model_args['return_model'])

    def forward(self, obs, action, adv_action=None):
        bsz, t = obs.shape[:2]
        if self.include_adv:
            x = torch.cat([obs, action, adv_action], dim=-1).view(bsz * t, -1)
        else:
            x = torch.cat([obs, action], dim=-1).view(bsz * t, -1)
        ret_obs_act_reps = self.ret_obs_action_model(x).view(bsz, t, -1)

        if self.is_lstm:
            # Use LSTM to get the representations for each suffix of the sequence
            hidden = (torch.zeros(1, bsz, self.hidden_dim).to(x.device), torch.zeros(1, bsz, self.hidden_dim).to(x.device))
            x, _ = self.lstm_model(ret_obs_act_reps, hidden)

            # Reverse the sequence in time again
            ret_pred = self.return_model(x.view(bsz, t, -1))
        else:
            ret_pred = self.return_model(ret_obs_act_reps).view(bsz, t, -1)
        return ret_pred


class RtgNetwork(torch.nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """
    def __init__(
        self,
        state_dim: int=1,
        action_dim: int=3,
        adv_action_dim: int=7,
        hidden_dim: int=512,
        include_adv: bool = False
    ):
        super().__init__()
        self.config = CustomNetworkConfig()
        self.state_dim = state_dim
        self.act_dim = action_dim
        self.adv_action_dim = adv_action_dim
        self.include_adv = include_adv

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.act_embed = torch.nn.Sequential(
            torch.nn.Linear(self.act_dim, hidden_dim), torch.nn.ReLU(),
        )

        if include_adv:
            self.adv_act_embed = torch.nn.Sequential(
                torch.nn.Linear(self.adv_action_dim, hidden_dim), torch.nn.ReLU(),
            )

        self.obs_embed = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, hidden_dim), torch.nn.ReLU(),
        )
        if include_adv:
            self.rtg_net = torch.nn.Sequential(
                torch.nn.Linear(3 * hidden_dim, hidden_dim), torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, 1)
            )
        else:
            self.rtg_net = torch.nn.Sequential(
                torch.nn.Linear(2 * hidden_dim, hidden_dim), torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, 1)
            )

    def forward(self, observation: torch.Tensor, pr_action, adv_action=None): 
        if not self.include_adv:
            act_emd, obs_emd = self.act_embed(pr_action), self.obs_embed(observation)
            return self.rtg_net(torch.cat([obs_emd, act_emd], dim=-1))
        else:
            act_emd, obs_emd, adv_emd = self.act_embed(pr_action), self.obs_embed(observation), self.adv_act_embed(adv_action)
            return self.rtg_net(torch.cat([obs_emd, act_emd, adv_emd], dim=-1))
 
 
class RtgLSTM(torch.nn.Module):
    def __init__(self, state_dim, action_dim, adv_action_dim, model_args, hidden_dim=64, include_adv=False, is_lstm=True) -> None:
        super().__init__()
        self.include_adv = include_adv
        self.is_lstm = is_lstm

        hidden_dim = model_args['ret_obs_act_model_args']['hidden_size']
        self.hidden_dim = hidden_dim

        input_dim = state_dim + action_dim + adv_action_dim if include_adv else state_dim + action_dim
        self.ret_obs_act_model_args = MLP(input_dim, hidden_dim, **model_args['ret_obs_act_model_args'])
        
        if is_lstm:
            self.lstm_model = torch.nn.LSTM(hidden_dim, hidden_dim,batch_first=True)
            self.ret_model_args = torch.nn.Linear(hidden_dim, 1)
        else:
            self.ret_model_args = MLP(hidden_dim, 1, **model_args['ret_model_args'])

    def forward(self, obs, action, adv_action=None):
        bsz, t = obs.shape[:2]
        if self.include_adv:
            x = torch.cat([obs, action, adv_action], dim=-1).view(bsz * t, -1)
        else:
            x = torch.cat([obs, action], dim=-1).view(bsz * t, -1)
        ret_obs_act_reps = self.ret_obs_act_model_args(x).view(bsz, t, -1)

        if self.is_lstm:
            # Use LSTM to get the representations for each suffix of the sequence
            hidden = (torch.zeros(1, bsz, self.hidden_dim).to(x.device), torch.zeros(1, bsz, self.hidden_dim).to(x.device))
            x, _ = self.lstm_model(ret_obs_act_reps, hidden)

            # Reverse the sequence in time again
            ret_pred = self.ret_model_args(x.view(bsz, t, -1))
        else:
            ret_pred = self.ret_model_args(ret_obs_act_reps).view(bsz, t, -1)
        return ret_pred
