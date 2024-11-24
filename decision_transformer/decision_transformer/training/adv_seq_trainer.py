import torch

from decision_transformer.decision_transformer.training.trainer import Trainer


class AdvSequenceTrainer(Trainer):
    def train_step(self):
        """
        Train an Adversarial DT model for one training step.
        """
        states, actions, adv_actions, rewards, dones, returns, timesteps, attention_mask = self._get_batch()
        action_targets = torch.clone(actions)

        _, action_preds, _ = self.model.forward(
            states, actions, adv_actions, rewards, returns, timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds_masked = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_targets_masked = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            action_preds_masked,
            action_targets_masked
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.gradients_clipper(self.model.parameters())
        self.optimizer.step()

        return loss.detach().cpu().item()
