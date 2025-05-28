"""
Behaviour-Cloning trainer for SimpleAgent → DictPolicy
------------------------------------------------------

 * single-threaded env (faster than subprocess when expert is cheap)
 * dataset collected once, then supervised SGD
 * resume support (policy + optimizer)
"""

from __future__ import annotations

import os, time, random, json
from typing import Callable, List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import gymnasium as gym
import wandb

from agents.simple_agent import SimpleAgent          # uses its own decoder!
from Training.FeatureExtractor import DictPolicy     # your SB3 policy class
from Training.ActionWrapper import ActionWrapper     # env action flattener
from core.env import GMSEnv
from core.round_state_generator import generate_round_state


# ────────────────────────────────────────────────────────────────────
# 0  helper – make **one** env identical to PPO training env
# ────────────────────────────────────────────────────────────────────
def make_env() -> gym.Env:
    round_states = {
        "garden": generate_round_state("garden")[0],
        "cross": generate_round_state("cross")[0],
        "default": generate_round_state("default")[0]
    }
    env = GMSEnv({"round_states": round_states, "max_episode_steps": 128})
    env = ActionWrapper(env)          # same wrapper as PPO
    return env


# ────────────────────────────────────────────────────────────────────
# 1  very small replay buffer stored in memory
# ────────────────────────────────────────────────────────────────────
class ExpertDataset(Dataset):
    def __init__(self) -> None:
        self.obs: List[Dict[str, Any]] = []
        self.act: List[np.ndarray]     = []   # flattened Box actions

    def add(self, obs: Dict[str, Any], act: np.ndarray) -> None:
        self.obs.append(obs)
        self.act.append(act)

    # torch-style API
    def __len__(self) -> int:             return len(self.obs)
    def __getitem__(self, i):             return self.obs[i], self.act[i]

    def remove_last_n(self, n: int) -> None:
        """Remove the last n items from the dataset."""
        if n <= 0:
            return
        if n > len(self.obs):
            n = len(self.obs)
        # Remove the last n items
        self.obs = self.obs[:-n]
        self.act = self.act[:-n]

# ────────────────────────────────────────────────────────────────────
# 2  behaviour-cloning experiment
# ────────────────────────────────────────────────────────────────────
class BCExperiment:
    def __init__(
        self,
        env_fn: Callable[[], gym.Env] = make_env,
        total_expert_steps: int = 1000_000,
        bc_epochs: int = 8,
        batch_size: int = 512,
        lr: float = 3e-4,
        device: str = "cpu",
        ckpt_path: str = "models/bc_policy.pt",
    ) -> None:
        self.env  = env_fn()
        self.expert = SimpleAgent()   # pos is overwritten
        self.policy = DictPolicy(
            self.env.observation_space,
            self.env.action_space,
            lr_schedule=lambda _: lr,
        ).to(device)

        self.opt      = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.ds       = ExpertDataset()
        self.steps    = total_expert_steps
        self.epochs   = bc_epochs
        self.bs       = batch_size
        self.device   = device
        self.ckpt     = ckpt_path

        self.wrun = wandb.init(
            project="gms-behavior-cloning",
            name=f"bc_simple_agent_{time.strftime('%Y%m%d_%H%M%S')}",
            config=dict(
                algo="BC",
                expert_steps=total_expert_steps,
                epochs=bc_epochs,
                batch_size=batch_size,
                lr=lr,
                device=device,
            ),
            resume="never",
            save_code=True,
            settings=wandb.Settings(
                start_method="thread"
            ),
            tags=["behavior_cloning", "training"],
            notes="Behavior cloning training of SimpleAgent"
        )

    # ── 2.1  collect dataset ─────────────────────────────────────
    @torch.no_grad()
    def collect(self) -> None:
        obs, _ = self.env.reset()
        collected, t0 = 0, time.time()
        tick = 0

        while collected < self.steps:
            # NB:  SimpleAgent expects **raw** obs; it decodes internally
            act_dict = self.expert.act(obs)

            # convert dict → flattened Box via wrapper
            act = self.env.reverse_action(act_dict)
            tick += 1

            # store
            self.ds.add(obs, act.copy())

            # step
            obs, reward, done, truncated, _ = self.env.step(act)
            if done:
                self.expert.reset()
                tick = 0
                obs, _ = self.env.reset()

            if truncated:
                self.expert.reset()
                # remove all observations created in this truncated run
                self.ds.remove_last_n(tick)
                collected -= tick
                tick = 0
                obs, _ = self.env.reset()

            collected += 1
            if collected % 10_000 == 0:
                wandb.log(
                    {
                        "data/collected": collected,
                        "data/fps": collected / (time.time() - t0),
                    }
                )

        print(f"[BC] dataset size: {len(self.ds):,}")

    # ── 2.2  supervised training ────────────────────────────────
    def train(self) -> None:
        loader = DataLoader(self.ds, batch_size=self.bs, shuffle=True)

        for epoch in range(1, self.epochs + 1):
            for i, (obs_batch, act_batch) in enumerate(loader, 1):
                # turn list-of-dict → dict-of-tensor  (keys identical)
                tensor_obs = obs_batch
                act_tensor = torch.as_tensor(np.stack(act_batch), dtype=torch.float32).to(self.device)

                dist = self.policy.get_distribution(tensor_obs)

                if i % 50 == 0:
                    print("dist: ", dist.distribution.mean)
                    print("act_tensor: ", act_tensor)

                logp = dist.log_prob(act_tensor)
                loss = -(logp.mean())

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()

                if i % 50 == 0:
                    wandb.log({
                        "bc/loss": loss.item(),
                        "bc/epoch": epoch,
                        "bc/iter": i,
                        "bc/learning_rate": self.opt.param_groups[0]["lr"],
                        "bc/batch_size": len(obs_batch),
                        "bc/grad_norm": torch.nn.utils.clip_grad_norm_(self.policy.parameters(), float("inf")),
                        "bc/logp_mean": logp.mean().item(),
                        "bc/logp_std": logp.std().item()
                    })

            # checkpoint each epoch
            torch.save(
                {
                    "policy":   self.policy.state_dict(),
                    "optim":    self.opt.state_dict(),
                    "epoch":    epoch,
                },
                self.ckpt,
            )
            print(f"[BC] saved checkpoint after epoch {epoch}")

    # ── 2.3  public entry point ─────────────────────────────────
    def run(self) -> DictPolicy:
        if os.path.exists(self.ckpt):
            ck = torch.load(self.ckpt, map_location=self.device)
            self.policy.load_state_dict(ck["policy"])
            self.opt.load_state_dict(ck["optim"])
            print(f"[BC] resumed from {self.ckpt} (epoch {ck['epoch']})")
        else:
            print("[BC] collecting expert roll-outs …")
            self.collect()
            print("[BC] training policy …")
            self.train()
            print("[BC] done ✓")

        self.policy.to("cpu")
        self.env.close()
        wandb.finish()
        return self.policy


# ────────────────────────────────────────────────────────────────────
# 3  quick CLI test
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bc = BCExperiment(device="cpu")   # switch to "mps" / "cuda" if available
    trained_policy = bc.run()
    trained_policy.save("models/bc_policy_final.zip")
