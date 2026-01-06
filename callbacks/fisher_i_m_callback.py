import pathlib as pl
from typing import List, Optional, Sequence, Union

import torch
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback


class FIM_callback(BaseCallback):
    """Computes the diagonal Fisher information matrix and trace of the critic."""

    def __init__(
        self,
        verbose: int = 0,
        step_interval: int = 100,
        plot_path: Optional[Union[str, pl.Path]] = None,
    ):
        super().__init__(verbose)
        self.step_interval = step_interval
        self.plot_path = pl.Path(plot_path) if plot_path is not None else None
        self.diag_fisher: Optional[Sequence[torch.Tensor]] = None
        self.trace_history: List[float] = []
        self.trace_timesteps: List[int] = []

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.diag_fisher = [
            torch.zeros_like(p, device=p.device)
            for p in self.model.policy.critic.parameters()
        ]
        self.trace_history = []
        self.trace_timesteps = []

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
                self.trace_history.append(fisher_trace)
                self.trace_timesteps.append(self.num_timesteps)
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
        if self.plot_path is not None and self.trace_history:
            self.save_trace_plot(self.plot_path)

    def save_trace_plot(self, path: Union[str, pl.Path]) -> None:
        """Save a plot of the Fisher trace across timesteps to the given path."""

        if not self.trace_history:
            logger.warning("no fisher trace values to plot; skipping plot save")
            return

        import matplotlib.pyplot as plt

        plot_path = pl.Path(path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 4))
        plt.plot(self.trace_timesteps, self.trace_history, label="Fisher trace")
        plt.xlabel("Timestep")
        plt.ylabel("Trace value")
        plt.title("Fisher Information Trace")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"saved fisher trace plot to {plot_path}")
