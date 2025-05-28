import time
import wandb
from stable_baselines3.common.callbacks import BaseCallback

class TimingCallback(BaseCallback):
    def _on_training_start(self) -> None:
        # nothing to do until the first rollout
        pass

    def _on_rollout_start(self) -> None:
        # mark the wall-clock time just before env stepping begins
        self._start_time = time.perf_counter()

    def _on_rollout_end(self) -> None:
        # compute how long the env (reset+step) took
        env_time = time.perf_counter() - self._start_time
        # total timesteps so far
        ts = self.num_timesteps

        # log *immediately* to W&B, with a proper step index
        wandb.log({
            "timing/env_step_sec": env_time,
        }, step=ts)

        # store env_time so we can subtract from total in update end
        self._env_time = env_time

    def _on_update_end(self) -> None:
        # full elapsed from rollout start to end of update
        total_time = time.perf_counter() - self._start_time
        rest_time  = total_time - getattr(self, "_env_time", 0.0)
        ts = self.num_timesteps

        wandb.log({
            "timing/rest_sec": rest_time,
        }, step=ts)

    def _on_step(self) -> bool:
        # SB3 requires this, but we don't need to do anything per step
        return True
