"""Train DDPG agent on Swimmer environment."""
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback

from callbacks.changing_env_callback import change_env
from callbacks.mean_return_callback import MeanReturnCallback
from callbacks.rFIM_callback import RFIM_callback


def main() -> None:
    """Train and save DDPG agent with callbacks."""
    # Create environment
    env = gym.make("Swimmer-v5", render_mode="rgb_array")

    # Initialize callbacks
    artifact_dir = Path(__file__).parent / "artifacts"
    callback_return = MeanReturnCallback()
    callback_fim = RFIM_callback(plot_path=artifact_dir / "rFIM_trace.png")
    callbacks: list[BaseCallback] = [
        callback_return,
        callback_fim,
        change_env(1500000, 10, (False, True)),
    ]

    # Load hyperparameters
    hyper_param_path = (
        Path(__file__).parent.parent / "agent" / "hyper_param.json"
    )
    with open(hyper_param_path) as f:
        hyper_params: dict[str, Any] = json.load(f)

    # Load pre-trained model
    model = DDPG.load("ddpg_swimmer_optimize", env=env)

    # Train model
    model.learn(total_timesteps=1500000, callback=callbacks)

    # Save trained model
    model.save("ddpg_swimmer_no_optimize_half")

    # Save visualizations and data
    callback_return.plot_returns(artifact_dir / "mean_return.png")

    mean_return_data: dict[str, Any] = {
        "episode_returns": callback_return.episode_returns,
        "mean_returns": callback_return.mean_returns,
    }
    with open(artifact_dir / "mean_return_data.json", "w") as f:
        json.dump(mean_return_data, f)

    rfim_data: dict[str, Any] = {
        "relative_trace_history": callback_fim.relative_trace_history,
        "trace_timesteps": callback_fim.trace_timesteps,
        "trace_history": callback_fim.trace_history,
    }
    with open(artifact_dir / "rFIM_trace_data.json", "w") as f:
        json.dump(rfim_data, f)

    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
