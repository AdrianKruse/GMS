import gymnasium as gym
import numpy as np
from gymnasium import spaces

class ActionWrapper(gym.ActionWrapper):
    """
    Flatten Dict action into a 4-float Box:
        [cmd_norm, x_norm, y_norm, id_norm]  ∈ [0, 1]^4

    Decoding:
        cmd = round(cmd_norm * (N_ACTIONS-1))
        x   = round(x_norm   * (W-1))
        y   = round(y_norm   * (H-1))
        id  = round(id_norm  * (N_TOWERS-1))
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.orig = env.action_space          # Dict

        # --- cache counts / bounds ----------
        self.n_cmd   = self.orig["action_type"].n
        self.x_min, self.y_min = self.orig["target_position"].low.astype(int)
        self.x_max, self.y_max = self.orig["target_position"].high.astype(int)
        self.w = self.x_max - self.x_min + 1
        self.h = self.y_max - self.y_min + 1

        self.n_tower = self.orig["target_id"].n
        # if there can be only one tower, avoid divide-by-zero later
        self.n_tower = max(self.n_tower, 1)

        # --- new Box space in [0,1] ----------
        self.action_space = spaces.Box(
            low  = np.zeros(4, dtype=np.float32),
            high = np.ones (4, dtype=np.float32),
            dtype=np.float32,
        )

    # ----------------------------------------------------------
    def action(self, a: np.ndarray) -> dict:
        """Map normalized floats → Dict expected by the core env."""
        a = np.clip(a, 0.0, 1.0)

        cmd_norm, x_norm, y_norm, id_norm = a

        cmd = int(round(cmd_norm * (self.n_cmd   - 1)))
        x   = int(round(x_norm  * (self.w        - 1))) + self.x_min
        y   = int(round(y_norm  * (self.h        - 1))) + self.y_min
        tid = int(round(id_norm * (self.n_tower  - 1)))

        return {
            "action_type":     cmd,
            "target_position": (x, y),
            "target_id":       tid,
        }

    # optional: for logging / debugging
    def reverse_action(self, act: dict) -> np.ndarray:
        # write this such that each field which is not present is set to 0
        if act.get("action_type") is None:
            cmd = 0
        else:
            cmd = act["action_type"] / (self.n_cmd - 1)

        if act.get("target_position") is None:
            x = 0
            y = 0
        else:
            x   = (act["target_position"][0] - self.x_min) / (self.w - 1)
            y   = (act["target_position"][1] - self.y_min) / (self.h - 1)

        if act.get("target_id") is None:
            tid = 0.0
        elif self.n_tower == 1:
            tid = 0.0  # or float(act["target_id"]) if needed, but denominator must be ≠ 0
        else:
            tid = act["target_id"] / (self.n_tower - 1)
        return np.array([cmd, x, y, tid], dtype=np.float32)

    def get_map_name(self):
        return self.env.map_name
    
    def get_round_state(self):
        return self.env.round_state