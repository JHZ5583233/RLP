from stable_baselines3.common.callbacks import BaseCallback
import backpack 

# 

class RFIM_callback(BaseCallback):
    def __init__(self, verbose=0, step_interval:int =100):
        super().__init__(verbose)
        self.step_interval = step_interval
        self.step = 0

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """




        pass


    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        if self.step % self.step_interval == 0 :
        # compute rFIM 

            # backpack.extensions.DiagGGNMC()
            pass


        self.step += 1 
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass