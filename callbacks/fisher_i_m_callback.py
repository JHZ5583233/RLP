import torch
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback


class FIM_callback(BaseCallback):
    """Computes the Diagnal Fischer information matrix and trace of the critic"""

    def __init__(self, verbose=0, step_interval: int = 100):
        super().__init__(verbose)
        self.step_interval = step_interval
        self.diag_fisher = None

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.diag_fisher = [
            torch.zeros_like(p, device=p.device)
            for p in self.model.policy.critic.parameters()
        ]

        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """

        if self.num_timesteps % self.step_interval == 0:
            # check to see if the replay buffer has enough data
            replay_buffer = self.model.replay_buffer
            if replay_buffer.size() >= self.model.batch_size:
                # compute rFIM
                batch = replay_buffer.sample(self.model.batch_size)

                critic = self.model.policy.critic
                critic.zero_grad()

                q_values = critic(batch.observations, batch.actions)[0]
                q_values.mean().backward()

                with torch.no_grad():
                    for i, p in enumerate(critic.parameters()):
                        if p.grad is not None:
                            self.diag_fisher[i] += p.grad.pow(2)

                fisher_vector = torch.cat([p.flatten() for p in self.diag_fisher])
                fisher_trace = fisher_vector.sum().item()
                self.logger.record("fisher/trace", fisher_trace)
                logger.info(
                    f"computed fisher trace on timestep {self.num_timesteps}: {fisher_trace}"
                )

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
