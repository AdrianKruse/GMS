import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class GridSetFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=128)

        # --- unpack H, W, C correctly ---
        h, w, c = observation_space["grid_map"].shape

        # --- CNN branch ---
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        pool_k = 4
        assert h % pool_k == 0 and w % pool_k == 0, (
            f"Grid ({h}×{w}) must be divisible by pool size {pool_k}"
        )
        self.pool = nn.AvgPool2d(kernel_size=pool_k, stride=pool_k)
        self.flat = nn.Flatten()

        # precompute flattened dim: 16 channels × (h/pool_k) × (w/pool_k)
        ph, pw = h // pool_k, w // pool_k
        self.grid_flat_dim = 16 * ph * pw

        # --- per-object (proj/tower) branch ---
        self.obj_lat = 16
        self.obj_mlp = nn.Sequential(
            nn.Linear(4, self.obj_lat),
            nn.ReLU(),
        )

        # --- vector branch ---
        vec_in = observation_space["vector_state"].shape[0]
        self.vec_lat = 16
        self.vec_mlp = nn.Sequential(
            nn.Linear(vec_in, self.vec_lat),
            nn.ReLU(),
        )

        # --- fusion layer ---
        fused_size = self.grid_flat_dim + self.obj_lat + self.obj_lat + self.vec_lat
        self.fusion = nn.Sequential(
            nn.Linear(fused_size, 128),
            nn.ReLU(),
        )

    def forward(self, obs):
        # 1) get grid, permute to (B, C, H, W)
        grid = obs["grid_map"].float().permute(0, 3, 1, 2)
        x = self.cnn(grid)               # (B,16,H,W)
        x = self.pool(x)                 # (B,16,H/4,W/4)
        g = self.flat(x)                 # (B, grid_flat_dim)

        # 2) projectiles set → (B,10,4) → MLP → (B,10,16) → mean → (B,16)
        proj = obs["nearest_projectiles"].view(g.size(0), 10, 4)
        proj_emb = self.obj_mlp(proj).mean(dim=1)

        # 3) towers set → same shape logic
        tw = obs["nearest_towers"].view(g.size(0), 5, 4)
        tw_emb = self.obj_mlp(tw).mean(dim=1)

        # 4) vector → (B,16)
        v = self.vec_mlp(obs["vector_state"].float())

        # 5) fuse and return (B,128)
        fused = torch.cat([g, proj_emb, tw_emb, v], dim=1)
        return self.fusion(fused)

class DictPolicy(ActorCriticPolicy):
    """Actor-Critic policy that uses DictFeatureExtractor as backbone."""

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=GridSetFeatureExtractor,
            features_extractor_kwargs={},  # default proj_dim_per_key=128
            net_arch=[dict(pi=[128, 64], vf=[128, 64])],
            activation_fn=nn.ReLU,
            **kwargs,
        )

    # forward() and _predict() are inherited and need no override.

# ---------------- example usage ----------------
# from stable_baselines3 import PPO
# env = ActionWrapper(make_env())  # your wrapped environment
# model = PPO(policy=DictPolicy, env=env, learning_rate=3e-4, n_steps=2048,
#             batch_size=64, n_epochs=10, device="cpu", verbose=1)
