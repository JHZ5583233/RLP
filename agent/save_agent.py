from callbacks.mean_return_callback import MeanReturnCallback
from callbacks.rFIM_callback import RFIM_callback
import gymnasium as gym

from pathlib import Path
from stable_baselines3 import DDPG
from json import load, dump


if __name__ == "__main__":
    env = gym.make('Swimmer-v5', render_mode="rgb_array")
    callback_return = MeanReturnCallback()
    callback_fim = RFIM_callback(plot_path=Path(__file__).parent / "artifacts" / "rFIM_trace.png")
    callbacks = [callback_return, callback_fim]
    hyper_params = load(open(Path(__file__).parent.parent / "agent" / "hyper_param.json", "r"))

    model = DDPG("MlpPolicy", env, verbose=1,
                 learning_rate=hyper_params["learning_rate"],
                 batch_size=hyper_params["batch_size"],
                 buffer_size=hyper_params["buffer_size"],
                 tau=hyper_params["tau"],
                 gamma=hyper_params["gamma"])
    model.learn(total_timesteps=1500000, callback=callbacks)
    model.save("ddpg_swimmer_no_optimize")
    callback_return.plot_returns(Path(__file__).parent / "artifacts" / "mean_return.png")
    # Save callback data
    with open(Path(__file__).parent / "artifacts" / "mean_return_data.json", "w") as f:
        dump({"episode_returns": callback_return.episode_returns, "\n mean_returns": callback_return.mean_returns}, f)
    with open(Path(__file__).parent / "artifacts" / "rFIM_trace_data.json", "w") as f:
        dump({"relative_trace_history": callback_fim.relative_trace_history, "\n trace_timesteps": callback_fim.trace_timesteps, "\n trace_history": callback_fim.trace_history}, f)
    env.close()
