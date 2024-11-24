import torch

from decision_transformer.decision_transformer.training.trainer import Trainer


class ActTrainer(Trainer):
    def train_step(self):
        """
        Train a Behavioural Cloning model for one training step.
        """
        states, actions, rewards, dones, returns, timesteps, attention_mask = self._get_batch()
        action_targets = torch.clone(actions)

        _, action_preds, _ = self.model.forward(
            states, actions, rewards, attention_mask=attention_mask, target_return=returns[:,0],
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)
        action_targets = action_targets[:,-1].reshape(-1, act_dim)
        loss = self.loss_fn(
            action_preds,
            action_targets
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()
