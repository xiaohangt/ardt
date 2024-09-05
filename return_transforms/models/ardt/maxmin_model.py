import torch

from return_transforms.models.basic.mlp import MLP


class RtgFFN(torch.nn.Module):
    """
    The RtgFFN (Return-to-Go Feed-Forward Network) is a neural network model designed to predict 
    return-to-go (RTG) values based on the current state and past actions.

    The model consists of separate embedding layers for the action, adversarial action (if applicable), 
    and observation (state). These embeddings are concatenated and passed through a feed-forward network 
    (`rtg_net`) to output a single RTG prediction.

    Key Features:
    - `act_embed`: Embeds the previous action.
    - `adv_act_embed`: Embeds the adversarial action (optional).
    - `obs_embed`: Embeds the current state/observation.
    - `rtg_net`: Predicts the return-to-go based on the concatenated embeddings.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        adv_action_dim: int,
        hidden_dim: int = 512,
        include_adv: bool = False
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = action_dim
        self.adv_action_dim = adv_action_dim
        self.include_adv = include_adv

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
    """
    The RtgLSTM (Return-to-Go Long Short-Term Memory) is a model designed to predict return-to-go (RTG) values 
    in a sequential decision-making setting. This model utilizes an LSTM to capture temporal dependencies in the data.
    
    Key Features:
    - If `is_lstm` is set to `True`, it uses an LSTM to model sequential data and predict RTG values. Otherwise, it uses an MLP.
    - For LSTM-based modeling, the representations are processed through an LSTM to capture temporal patterns, 
    followed by a final linear layer to predict the RTG values.
    - If `is_lstm` is `False`, the MLP directly predicts RTG values based on the processed inputs.

    This design allows the model to handle both sequential data with temporal dependencies (via LSTM) and simpler cases 
    (via MLP), making it flexible for different types of return prediction tasks.
    """
    def __init__(
            self, 
            state_dim: int,
            action_dim: int,
            adv_action_dim: int, 
            model_args: dict, 
            hidden_dim: int = 64, 
            include_adv: bool = False, 
            is_lstm: bool = True
        ) -> None:
        super().__init__()
        self.include_adv = include_adv
        self.is_lstm = is_lstm

        hidden_dim = model_args['ret_obs_act_model_args']['hidden_size']
        self.hidden_dim = hidden_dim

        input_dim = state_dim + action_dim + adv_action_dim if include_adv else state_dim + action_dim
        self.ret_obs_act_model = MLP(input_dim, hidden_dim, **model_args['ret_obs_act_model_args'])
        
        if is_lstm:
            self.lstm_model = torch.nn.LSTM(hidden_dim, hidden_dim,batch_first=True)
            self.ret_model = torch.nn.Linear(hidden_dim, 1)
        else:
            self.ret_model = MLP(hidden_dim, 1, **model_args['ret_model_args'])

    def forward(
            self, 
            obs: torch.Tensor, 
            action: torch.Tensor, 
            adv_action: torch.Tensor | None = None
        ):
        # Concatenate observations and actions for recurrence
        batch_size, seq_len = obs.shape[:2]
        if self.include_adv:
            x = torch.cat([obs, action, adv_action], dim=-1).view(batch_size * seq_len, -1)
        else:
            x = torch.cat([obs, action], dim=-1).view(batch_size * seq_len, -1)
        
        ret_obs_act_reps = self.ret_obs_act_model(x).view(batch_size, seq_len, -1)
        if self.is_lstm:
            # Use LSTM to get the representations for each suffix of the sequence
            hidden = (
                torch.zeros(1, batch_size, self.hidden_dim).to(x.device), 
                torch.zeros(1, batch_size, self.hidden_dim).to(x.device)
            )
            x, _ = self.lstm_model(ret_obs_act_reps, hidden)

            # Reverse the sequence in time again and project through linear layer
            ret_pred = self.ret_model(x.view(batch_size, seq_len, -1))
        else:
            # Pass through MLP to get return prediction
            ret_pred = self.ret_model(ret_obs_act_reps).view(batch_size, seq_len, -1)
        
        return ret_pred
