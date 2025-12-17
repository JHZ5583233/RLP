import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np
import optuna
import pathlib as pl

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy


from json import dump


def objective(trial):
    train_env = gym.make("Swimmer-v5")

    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )

    model = DDPG(
        "MlpPolicy",
        train_env,
        learning_rate=trial.suggest_float("learning_rate",
                                          1e-4, 1e-2,
                                          log=True),
        buffer_size=trial.suggest_int("buffer_size",
                                      300_000, 2_000_000,
                                      step=100_000),
        batch_size=trial.suggest_categorical("batch_size",
                                             [128, 256, 512]),
        tau=trial.suggest_float("tau",
                                1e-4, 0.02,
                                log=True),
        gamma=trial.suggest_float("gamma",
                                  0.90, 0.999),
        action_noise=action_noise,
        verbose=0
    )

    model.learn(total_timesteps=5000,
                progress_bar=True)

    eval_env = gym.make("Swimmer-v5")
    eval_env = TimeLimit(eval_env, max_episode_steps=500)
    print("evaluating")
    mean_reward, _ = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=5,
        deterministic=True
    )

    return mean_reward


def create_param():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective,
                   n_trials=40,
                   n_jobs=10,
                   show_progress_bar=True)

    print(study.best_params)
    with open(pl.Path(__file__).parent.joinpath("hyper_param.json"),
              "w+") as j:
        dump(study.best_params, j, indent=4)


if __name__ == '__main__':
    create_param()
