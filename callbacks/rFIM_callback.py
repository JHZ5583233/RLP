"""Relative Fisher Information Metric callback for Stable-Baselines3."""

import pathlib as pl
from typing import List, Optional, Sequence, Union

import torch
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback


class RFIM_callback(BaseCallback):
    """Computes a relative Fisher trace (incremental change) and can plot it."""

    def __init__(
        self,
        verbose: int = 0,
        step_interval: int = 100,
        plot_path: Optional[Union[str, pl.Path]] = None,
    ) -> None:
        super().__init__(verbose)
        self.step_interval = step_interval
        self.plot_path = pl.Path(plot_path) if plot_path is not None else None

        self.diag_fisher: Optional[Sequence[torch.Tensor]] = None
        self.prev_diag_fisher: Optional[Sequence[torch.Tensor]] = None
        self.trace_history: List[float] = []
        self.relative_trace_history: List[float] = []
        self.trace_timesteps: List[int] = []

    def _on_training_start(self) -> None:
        """Allocate storage and reset histories."""

        self.diag_fisher = [
            torch.zeros_like(p, device=p.device)
            for p in self.model.policy.critic.parameters()
        ]
        self.prev_diag_fisher = [p.clone() for p in self.diag_fisher]

        self.trace_history = []
        self.relative_trace_history = []
        self.trace_timesteps = []

    def _on_step(self) -> bool:
        """Compute and log relative Fisher trace at the configured interval."""

        if self.num_timesteps % self.step_interval == 0:
            replay_buffer = self.model.replay_buffer
            if replay_buffer.size() >= self.model.batch_size:
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

                relative_vector = torch.cat(
                    [
                        (self.diag_fisher[i] - self.prev_diag_fisher[i]).flatten()
                        for i in range(len(self.diag_fisher))
                    ]
                )
                relative_trace = relative_vector.clamp_min(0).sum().item()

                self.trace_history.append(fisher_trace)
                self.relative_trace_history.append(relative_trace)
                self.trace_timesteps.append(self.num_timesteps)

                # log to SB3 and to stdout
                self.logger.record("fisher/trace", fisher_trace)
                self.logger.record("fisher/trace_relative", relative_trace)
                logger.info(
                    f"computed fisher trace (abs {fisher_trace:.4f}, relative {relative_trace:.4f}) on timestep {self.num_timesteps}"
                )

                # update reference for next relative computation
                self.prev_diag_fisher = [p.clone() for p in self.diag_fisher]

        return True

    def _on_rollout_end(self) -> None:
        """No-op hook kept for symmetry."""
        return None

    def _on_training_end(self) -> None:
        """Optionally persist a plot of the relative Fisher trace."""

        if self.plot_path is not None and self.relative_trace_history:
            self.save_trace_plot(self.plot_path)

    def save_trace_plot(self, path: Union[str, pl.Path]) -> None:
        """Save a plot of relative (and absolute) Fisher traces to ``path``."""

        if not self.relative_trace_history:
            logger.warning("no relative fisher trace values to plot; skipping plot save")
            return

        import matplotlib.pyplot as plt

        plot_path = pl.Path(path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 4))
        plt.plot(self.trace_timesteps, self.relative_trace_history, label="Relative trace")
        if self.trace_history:
            plt.plot(self.trace_timesteps, self.trace_history, label="Absolute trace", alpha=0.6)
        plt.xlabel("Timestep")
        plt.ylabel("Trace value")
        plt.title("Relative Fisher Information Trace")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"saved relative fisher trace plot to {plot_path}")