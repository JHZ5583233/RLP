from stable_baselines3.common.callbacks import BaseCallback
from copy import copy


class change_env(BaseCallback):
    def __init__(self,
                 period_of_change: int = 1000,
                 amount_change: float = 2.0,
                 change_viscosity_density: tuple[bool, bool] = (False, False),
                 verbose: int = 0):
        super().__init__(verbose)
        self.period = period_of_change
        self.change = amount_change
        self.to_change = change_viscosity_density

        self.step = 0

    def _on_training_start(self) -> None:
        self.env = self.training_env.envs[0].unwrapped
        self.model_mujoco = self.env.model

        self.init_viscosity = copy(self.model_mujoco.opt.viscosity)
        self.init_density = copy(self.model_mujoco.opt.density)

        self.step = 0

    def _on_step(self) -> bool:
        done = self.locals.get("dones", [False])[0]
        if done:
            self.step = 0

        self.step += 1
        ratio = min(self.step/self.period, 1)

        if self.to_change[0]:
            new_val = self.change * self.init_viscosity
            self.model_mujoco.opt.viscosity = (
                (ratio * (new_val - self.init_viscosity)) +
                self.init_viscosity)

        if self.to_change[1]:
            new_val = self.change * self.init_density
            self.model_mujoco.opt.density = (
                (ratio * (new_val - self.init_density)) +
                self.init_density)

        if self.verbose > 0 and self.step % 100 == 0:
            print(
                f"Step {self.step}: " +
                f"viscosity = {self.model_mujoco.opt.viscosity} " +
                f"density = {self.model_mujoco.opt.density}"
            )

        return True


def main():
    import gymnasium as gym

    from stable_baselines3 import DDPG
    from stable_baselines3.common.noise import NormalActionNoise

    env = gym.make('Swimmer-v5', render_mode="rgb_array")

    model = DDPG("MlpPolicy",
                 env,
                 verbose=1)
    model.learn(total_timesteps=10000,
                log_interval=10,
                callback=change_env(verbose=1,
                                    change_viscosity_density=(True, True)))


if __name__ == '__main__':
    main()
