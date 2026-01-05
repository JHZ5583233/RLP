from callbacks.mean_return_callback import MeanReturnCallback
import gymnasium as gym

from pathlib import Path
from stable_baselines3 import DDPG
from json import load


if __name__ == "__main__":
    env = gym.make('Swimmer-v5', render_mode="rgb_array")
    callback = MeanReturnCallback()
    hyper_params = load(open(Path(__file__).parent.parent / "agent" / "hyper_param.json", "r"))
    # Continue training from a saved model
    model = DDPG("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=2000000, callback=callback)
    model.save("ddpg_swimmer_no_optimize")