import torch


class MLPBCModel(torch.nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) model for Behavior Cloning (BC).
    The model predicts the next action based on past states.

    Args:
        state_dim (int): Dimensionality of the state space.
        act_dim (int): Dimensionality of the action space.
        hidden_size (int): Size of the hidden layers.
        n_layer (int): Number of hidden layers in the MLP.
        dropout (float): Dropout probability. Default is 0.1.
        max_length (int): Maximum sequence length of states. Default is 1.
        **kwargs: Additional keyword arguments for flexibility (not used explicitly).
    """

    def __init__(
            self, 
            state_dim: int, 
            act_dim: int, 
            hidden_size: int, 
            n_layer: int, 
            dropout: float = 0.1, 
            max_length: int = 1, 
            **kwargs
        ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.hidden_size = hidden_size

        # Define the MLP layers
        # First layer projects the input (state) to the hidden size dimension
        layers = [torch.nn.Linear(max_length * self.state_dim, hidden_size)]
        
        # Add hidden layers with ReLU activations and dropout
        for _ in range(n_layer - 1):
            layers.extend([
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_size, hidden_size)
            ])
        
        # Output layer projects to the action dimension with Tanh activation
        layers.extend([
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, self.act_dim),
            torch.nn.Tanh(),
        ])

        # Combine layers into a sequential model
        self.model = torch.nn.Sequential(*layers)

    def forward(
            self, 
            states: torch.Tensor, 
            actions: torch.Tensor, 
            rewards: torch.Tensor, 
            attention_mask: torch.Tensor | None = None, 
            target_return: torch.Tensor | None = None
        ):
        """
        Args:
            states (torch.Tensor): The input state tensor (batch_size, sequence_length, state_dim).
            actions (torch.Tensor): Not used, included for compatibility with other methods.
            rewards (torch.Tensor): Not used, included for compatibility with other methods.
            attention_mask (torch.Tensor, optional): Not used, included for compatibility.
            target_return (torch.Tensor, optional): Not used, included for compatibility.

        Returns:
            tuple: (None, predicted actions, None)
        """
        # Limit the input states to the last `max_length` states and flatten them
        states = states[:, -self.max_length:].reshape(states.shape[0], -1)

        # Pass the flattened states through the model to get the predicted action
        actions = self.model(states).reshape(states.shape[0], 1, self.act_dim)
        
        # Return None for unused outputs and the predicted actions
        return None, actions, None

    def get_action(
            self, 
            states: torch.Tensor, 
            actions: torch.Tensor, 
            rewards: torch.Tensor, 
            **kwargs
        ):
        """
        Args:
            states (torch.Tensor): The input state tensor (sequence_length, state_dim).
            actions (torch.Tensor): Not used, included for compatibility.
            rewards (torch.Tensor): Not used, included for compatibility.

        Returns:
            torch.Tensor: The predicted action tensor (1, act_dim).
        """
        # Reshape the input state to match the expected input format (1, sequence_length, state_dim)
        states = states.reshape(1, -1, self.state_dim).to(dtype=torch.float32)
        
        # If the sequence is shorter than max_length, pad it with zeros
        if states.shape[1] < self.max_length:
            zeros = torch.zeros(
                (1, self.max_length - states.shape[1], self.state_dim), 
                dtype=torch.float32, 
                device=states.device
            )
            states = torch.cat([zeros, states], dim=1)

        # Run a forward pass to predict the action
        _, actions, _ = self.forward(states, None, None, **kwargs)
        
        # Return the predicted action for the last state in the sequence
        return actions[0, -1]
