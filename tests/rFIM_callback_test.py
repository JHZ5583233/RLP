import pathlib as pl
import sys
from json import load

import gymnasium as gym
from stable_baselines3 import DDPG

# Ensure project root is importable when running directly
PROJECT_ROOT = pl.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from callbacks.rFIM_callback import RFIM_callback


def rfim_callback_test():
    env = gym.make("Swimmer-v5", render_mode="rgb_array")
    plot_path = pl.Path(__file__).parent / "artifacts" / "rfisher_trace.png"
    callback = RFIM_callback(plot_path=plot_path)
    parameters_path = PROJECT_ROOT / "agent" / "hyper_param.json"
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
        verbose=1,
    )
    model.learn(total_timesteps=1000, callback=callback)
    env.close()
    return plot_path


if __name__ == "__main__":
    rfim_callback_test()
