from callbacks.mean_return_callback import MeanReturnCallback
from callbacks.fisher_i_m_callback import FIM_callback
import gymnasium as gym

from pathlib import Path
from stable_baselines3 import DDPG
from json import load, dump


if __name__ == "__main__":
    env = gym.make('Swimmer-v5', render_mode="rgb_array")
    callback = MeanReturnCallback()
    fim_callback = FIM_callback()
    callbacks = [callback, fim_callback]
    hyper_params = load(open(Path(__file__).parent.parent / "agent" / "hyper_param.json", "r"))
    # Continue training from a saved model
    model = DDPG("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=5000, callback=callbacks)
    # model.save("ddpg_swimmer_no_optimize")
    # Save mean returns to a json file
    with open("mean_returns.json", "w") as f:
        dump(callback.mean_returns, f)
    with open("fisher_information_matrix.json", "w") as f:
        dump(fim_callback.traces, f)
