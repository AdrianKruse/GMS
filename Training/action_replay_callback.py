import json
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class ActionReplayCallback(BaseCallback):
    """
    Collect (seed, action-sequence) per episode.
    Every `save_every` episodes dump a JSON file to `out_dir`.
    Works for single env (N_ENVS=1) or tracks only the first environment in VecEnv.
    """
    def __init__(self, save_every: int, out_dir: str = "replays", verbose=0):
        super().__init__(verbose)
        self.save_every = save_every
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._ep_actions = []      # rolling list for current episode
        self._pending = []         # list[dict] to dump next time
        self._current_seed = None  # store seed from episode start

    def _on_rollout_start(self) -> None:
        """Called at the start of each rollout - grab initial seed."""
        if self.verbose >= 1:
            print(f"[ActionReplayCallback] Starting new rollout at timestep {self.num_timesteps}")

    def _on_step(self) -> bool:
        """Called after each environment step."""
        # Access the local variables from PPO's collect_rollouts
        if 'actions' not in self.locals or 'dones' not in self.locals or 'infos' not in self.locals:
            return True  # Skip if variables not available
            
        actions = self.locals["actions"]  # shape: (n_envs,) 
        dones = self.locals["dones"]      # shape: (n_envs,)
        infos = self.locals["infos"]      # list of dicts, length n_envs
        
        # Only track the first environment (index 0)
        env_idx = 0
        
        if env_idx < len(infos):
            action = actions[env_idx] if hasattr(actions, '__getitem__') else actions
            done = dones[env_idx] if hasattr(dones, '__getitem__') else dones
            info = infos[env_idx]
            
            # Store the seed at the start of an episode
            if self._current_seed is None and 'seed' in info:
                self._current_seed = int(info['seed'])
                if self.verbose >= 2:
                    print(f"[ActionReplayCallback] New episode started with seed: {self._current_seed}")
            
            # Record the action
            if hasattr(action, 'item'):
                action_val = int(action.item())  # Convert tensor to int
            else:
                action_val = int(action)
            self._ep_actions.append(action_val)
            
            # If episode finished, save it
            if done:
                if self._current_seed is not None:
                    episode_data = {
                        "seed": self._current_seed,
                        "actions": self._ep_actions.copy(),
                        "episode_length": len(self._ep_actions),
                        "timestep": self.num_timesteps
                    }
                    self._pending.append(episode_data)
                    
                    if self.verbose >= 1:
                        print(f"[ActionReplayCallback] Completed episode with seed {self._current_seed}, "
                              f"{len(self._ep_actions)} actions")
                
                # Reset for next episode
                self._ep_actions = []
                self._current_seed = None
                
                # Check if we should dump
                if len(self._pending) >= self.save_every:
                    self._dump()

        return True

    def _dump(self):
        """Save collected episodes to JSON file."""
        if not self._pending:
            return
            
        fname = self.out_dir / f"rollouts_{self.num_timesteps}.json"
        data = {
            "metadata": {
                "total_episodes": len(self._pending),
                "created_at_timestep": self.num_timesteps,
                "format_version": "1.0"
            },
            "episodes": self._pending
        }
        
        with fname.open("w") as f:
            json.dump(data, f, indent=2)
            
        if self.verbose >= 1:
            print(f"[ActionReplayCallback] Saved {len(self._pending)} episodes to {fname}")
        self._pending.clear()

    def _on_training_end(self) -> None:
        """Save any remaining episodes when training ends."""
        if self._pending:
            if self.verbose >= 1:
                print(f"[ActionReplayCallback] Training ended, saving {len(self._pending)} remaining episodes")
            self._dump()
