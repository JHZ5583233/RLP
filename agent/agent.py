import gymnasium as gym
import numpy as np
import optuna
import pathlib as pl

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise


from json import dump


def objective(trial):
    env = gym.make('Swimmer-v5', render_mode="rgb_array")

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                     sigma=0.1 * np.ones(n_actions))

    learning_rate = trial.suggest_float("learning_rate",
                             1e-4, 1e-2,
                             log=True)
    buffer_size = trial.suggest_int("buffer_size",
                                    100000, 2_000_000,
                                    step=100000)
    batch_size = trial.suggest_categorical("batch_size",
                                           [64, 128, 256, 512])
    tau = trial.suggest_float("tau",
                              1e-4, 0.02,
                              log=True)
    gamma = trial.suggest_float("gamma",
                                0.90, 0.999)


    model = DDPG("MlpPolicy",
                 env,
                 learning_rate=learning_rate,
                 buffer_size=buffer_size,
                 batch_size=batch_size,
                 tau=tau,
                 gamma=gamma,
                 action_noise=action_noise,
                 verbose=1)
    model.learn(total_timesteps=1000, log_interval=10)
    vec_env = model.get_env()

    obs = vec_env.reset()
    cumulative_reward = 0
    for _ in range(999):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        cumulative_reward += rewards

    return cumulative_reward


def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=250)

    print(study.best_params)
    with open(pl.Path(__file__).parent.joinpath("hyper_param.json"),
              "w+") as j:
        dump(study.best_params, j, indent=4)


if __name__ == '__main__':
    main()
