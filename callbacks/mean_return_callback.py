from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from loguru import logger

class MeanReturnCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MeanReturnCallback, self).__init__(verbose)
        self.returns = []
        self.mean_returns = []
        self.episode_counts = 0

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
        self.episode_counts += 1
        logger.info(f"Episode {self.episode_counts} - Length: {episode_length}, Return: {episode_return:.2f}, Mean Return: {mean_return:.2f}")
    
    def _on_training_end(self):
        logger.info("Training ended.")