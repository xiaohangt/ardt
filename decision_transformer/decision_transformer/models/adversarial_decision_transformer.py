import torch
import transformers
from decision_transformer.decision_transformer.models.trajectory_gpt2 import GPT2Model


class AdversarialDecisionTransformer(torch.nn.Module):
    """
    Predicts next step actions based on past states, actions, and adversarial actions,
    conditional on the returns-to-go.
    
    Args:
        state_dim (int): Dimension of the state space.
        act_dim (int): Dimension of the action space.
        adv_act_dim (int): Dimension of the adversarial action space.
        hidden_size (int): Size of the hidden layers.
        max_length (int, optional): Maximum sequence length of states. Default is None.
        max_ep_len (int): Maximum episode length. Default is 4096.
        action_tanh (bool): Whether to apply Tanh activation to actions. Default is True.
        rtg_seq (bool): Whether to use returns-to-go in sequence. Default is True.
        **kwargs: Additional keyword arguments for the transformer configuration.
    """
    
    def __init__(
            self,
            state_dim: int,
            act_dim: int,
            adv_act_dim: int,
            hidden_size: int,
            max_length: int | None = None,
            max_ep_len: int = 4096,
            action_tanh: bool = True,
            rtg_seq: bool = True,
            **kwargs
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.adv_act_dim = adv_act_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.rtg_seq = rtg_seq
        
        # Initialize GPT2 transformer model configuration
        config = transformers.GPT2Config(
            vocab_size=1,  # No vocabulary is used in this task
            n_embd=hidden_size,
            **kwargs
        )
        self.transformer = GPT2Model(config)

        # Embedding layers for different inputs
        self.embed_timestep = torch.nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        self.embed_adv_action = torch.nn.Linear(self.adv_act_dim, hidden_size)
        self.embed_ln = torch.nn.LayerNorm(hidden_size)

        # Output layers
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        if self.rtg_seq:
            self.predict_action = torch.nn.Sequential(
                *([torch.nn.Linear(hidden_size, self.act_dim)] + ([torch.nn.Tanh()] if action_tanh else []))
            )
        else:
            self.predict_action = torch.nn.Sequential(
                *([
                    torch.nn.Linear(hidden_size * 2, hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_size, hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_size, self.act_dim)
                ] + ([torch.nn.Tanh()] if action_tanh else []))
            )
        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        adv_actions: torch.Tensor,
        rewards: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            states (torch.Tensor): Input states (batch_size, seq_length, state_dim).
            actions (torch.Tensor): Input actions (batch_size, seq_length, act_dim).
            adv_actions (torch.Tensor): Input adversarial actions (batch_size, seq_length, adv_act_dim).
            rewards (torch.Tensor): Input rewards (not used in the forward pass).
            returns_to_go (torch.Tensor): Input returns-to-go (batch_size, seq_length, 1).
            timesteps (torch.Tensor): Input timesteps (batch_size, seq_length).
            attention_mask (torch.Tensor, optional): Attention mask for the transformer. Default is None.

        Returns:
            tuple: Predicted states, actions, and returns.
        """
        embed_per_timestep = 4 if self.rtg_seq else 3
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # Default attention mask
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # Embedding inputs
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        adv_action_embeddings = self.embed_adv_action(adv_actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # Add time embeddings to states, actions, and returns
        state_embeddings += time_embeddings
        action_embeddings += time_embeddings
        adv_action_embeddings += time_embeddings
        if self.rtg_seq:
            returns_embeddings += time_embeddings

        # Stack the inputs in the autoregressive sequence order
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings, adv_action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, embed_per_timestep * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Adjust attention mask for the stacked inputs
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, embed_per_timestep * seq_length)

        # Transformer forward pass
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # Reshape output to match the original sequence structure
        x = x.reshape(batch_size, seq_length, embed_per_timestep, self.hidden_size).permute(0, 2, 1, 3)

        # Predict next states, actions, and returns
        return_preds = self.predict_return(x[:, 2])       # Predict next return
        state_preds = self.predict_state(x[:, 2])         # Predict next state
        action_preds = self.predict_action(x[:, 1])       # Predict next action

        return state_preds, action_preds, return_preds

    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        adv_actions: torch.Tensor,
        rewards: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        batch_size: int,
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            states (torch.Tensor): Input states (batch_size, seq_length, state_dim).
            actions (torch.Tensor): Input actions (batch_size, seq_length, act_dim).
            adv_actions (torch.Tensor): Input adversarial actions (batch_size, seq_length, adv_act_dim).
            rewards (torch.Tensor): Input rewards (not used).
            returns_to_go (torch.Tensor): Input returns-to-go (batch_size, seq_length, 1).
            timesteps (torch.Tensor): Input timesteps (batch_size, seq_length).
            batch_size (int): Batch size.

        Returns:
            torch.Tensor: Predicted next action.
        """
        # Reshape inputs
        states = states.reshape(batch_size, -1, self.state_dim)
        actions = actions.reshape(batch_size, -1, self.act_dim)
        adv_actions = adv_actions.reshape(batch_size, -1, self.adv_act_dim)
        returns_to_go = returns_to_go.reshape(batch_size, -1, 1)
        timesteps = timesteps.reshape(batch_size, -1)

        # Truncate sequences to max_length and pad if needed
        attention_mask = None

        if self.max_length is not None:
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            adv_actions = adv_actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]

            # Pad all tokens to sequence length
            attention_mask = torch.cat([
                torch.zeros((batch_size, self.max_length - states.shape[1])),
                torch.ones((batch_size, states.shape[1]))
            ], dim=1)
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)

            # Pad sequences
            states = torch.cat([
                torch.zeros((states.shape[0], self.max_length - states.shape[1], self.state_dim), device=states.device), 
                states
            ], dim=1).to(dtype=torch.float32)

            actions = torch.cat([
                torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim), device=actions.device), 
                actions
            ], dim=1).to(dtype=torch.float32)

            adv_actions = torch.cat([
                torch.zeros((adv_actions.shape[0], self.max_length - adv_actions.shape[1], self.adv_act_dim), device=adv_actions.device), 
                adv_actions
            ], dim=1).to(dtype=torch.float32)

            returns_to_go = torch.cat([
                torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device),
                returns_to_go
            ], dim=1).to(dtype=torch.float32)

            timesteps = torch.cat([
                torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), 
                timesteps
            ], dim=1).to(dtype=torch.long)

        # Get predictions and return
        _, action_preds, _ = self.forward(
            states, actions, adv_actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs
        )

        return action_preds
