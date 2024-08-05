import torch

from decision_transformer.decision_transformer.models.model import TrajectoryModel


class MLPBCModel(TrajectoryModel):
    """
    Simple MLP that predicts next action a from past states s.
    """
    def __init__(self, state_dim, act_dim, hidden_size, n_layer, dropout=0.1, max_length=1, **kwargs):
        super().__init__(state_dim, act_dim)
        self.hidden_size = hidden_size
        self.max_length = max_length

        layers = [torch.nn.Linear(max_length*self.state_dim, hidden_size)]
        for _ in range(n_layer-1):
            layers.extend([
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_size, hidden_size)
            ])
        layers.extend([
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, self.act_dim),
            torch.nn.Tanh(),
        ])

        self.model = torch.nn.Sequential(*layers)

    def forward(self, states, actions, rewards, attention_mask=None, target_return=None):
        states = states[:,-self.max_length:].reshape(states.shape[0], -1)
        actions = self.model(states).reshape(states.shape[0], 1, self.act_dim)
        return None, actions, None

    def get_action(self, states, actions, rewards, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        if states.shape[1] < self.max_length:
            zeros = (
                torch.zeros((1, self.max_length-states.shape[1], self.state_dim), dtype=torch.float32, device=states.device)
            )
            states = torch.cat([zeros, states], dim=1)
        states = states.to(dtype=torch.float32)
        _, actions, _ = self.forward(states, None, None, **kwargs)
        return actions[0, -1]
