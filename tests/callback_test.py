import gymnasium as gym
from callbacks.mean_return_callback import MeanReturnCallback
import pathlib as pl

from stable_baselines3 import DDPG

from json import load

env = gym.make('Swimmer-v5', render_mode="rgb_array")
callback = MeanReturnCallback()
parameters_path = pl.Path(__file__).parent.parent / "agent" / "hyper_param.json"
with open(parameters_path, "r") as f:
    hyper_params = load(f)

model = DDPG(
    "MlpPolicy",
    env,
    learning_rate=hyper_params["learning_rate"],
    gamma=hyper_params["gamma"],
    tau=hyper_params["tau"],
    buffer_size=hyper_params["buffer_size"],
    batch_size=hyper_params["batch_size"],
    verbose=1
)
model.learn(total_timesteps=100000, callback=callback)