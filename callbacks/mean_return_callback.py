import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from loguru import logger
import matplotlib.pyplot as plt

class MeanReturnCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MeanReturnCallback, self).__init__(verbose)
        self.returns = []
        self.mean_returns = []
        self.episode_counts = 0
        self.episode_returns = []  # Track actual episode returns

    def _on_training_start(self):
        self.episode_counts = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get('rewards')
        dones = self.locals.get('dones')

        self.returns.append(rewards.item() if isinstance(rewards, np.ndarray) else rewards)

        if np.any(dones):
            self._on_episode_end()
        return True
    
    def _on_episode_end(self):
        episode_length = len(self.returns)
        episode_return = sum(self.returns)
        self.returns = []
        mean_return = episode_return / episode_length if episode_length > 0 else 0.0
        self.mean_returns.append(mean_return)
        self.episode_returns.append(episode_return)
        self.episode_counts += 1
        logger.info(f"Episode {self.episode_counts} - Length: {episode_length}, Return: {episode_return:.2f}, Mean Return: {mean_return:.2f}")
    
    def _on_training_end(self):
        logger.info("Training ended.")
    
    def get_return_trace(self):
        """Returns the trace of returns for plotting."""
        return self.episode_returns.copy()
    
    def get_mean_return_trace(self):
        """Returns the trace of mean returns for plotting."""
        return self.mean_returns.copy()
    
    def plot_returns(self, save_path=None, show=True):
        """Plot the return trace."""
        fig, ax = plt.subplots(figsize=(10, 6))
        episodes = list(range(1, len(self.episode_returns) + 1))
        ax.plot(episodes, self.episode_returns, label='Episode Returns', alpha=0.6)
        ax.plot(episodes, self.mean_returns, label='Mean Returns', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Return')
        ax.set_title('Training Return Trace')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Return trace saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig, ax