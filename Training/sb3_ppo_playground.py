import inspect
import time
import gymnasium as gym
import numpy as np
import os
import torch
import wandb
import random
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor # Recommended for logging episode rewards
from wandb.integration.sb3 import WandbCallback # The crucial import!
from Training.ActionWrapper import ActionWrapper
from stable_baselines3.common.torch_layers import FlattenExtractor
from Training.FeatureExtractor import DictPolicy
from Training.timing_callback import TimingCallback

from core.round_state_generator import generate_round_state

from core.round_logic.state import RoundState
from core.state.game_state import GameState
from core.adapters.controller_adapter import ControllerAdapter
from core.state.game_state import Tower

# --- Your Environment Constants ---
GRID_WIDTH = 16
GRID_HEIGHT = 16
# ... other env constants ...

# Assuming your custom environment is in my_game_env/env.py
from core.env import GMSEnv

# --- Training Configuration ---
TOTAL_TIMESTEPS = 100_000
N_ENVS = 16
# You won't use tensorboard_log for the model directly if using W&B sync_tensorboard=True
# But W&B can still sync if SB3 is set to log to TB first, or use the callback directly.

### --- W&B Initialization ---
# Define a config dictionary for your W&B run.
# This will be logged as hyperparameters in your W&B dashboard.
wandb_config = {
    "policy_type": "MultiInputPolicy",
    "total_timesteps": TOTAL_TIMESTEPS,
    "env_name": "GMSEnv", # A descriptive name for your environment
    "n_envs": N_ENVS,
    "learning_rate": 0.0003,
    "n_steps": 1024,
    "batch_size": 256,
    "n_epochs": 5,
    "gamma": 0.98,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "net_arch": dict(pi=[64, 64], vf=[64, 64]),
    "activation_fn": "ReLU" # Store as string if not using direct torch.nn.ReLU object
}


# --- Environment Setup ---
def make_env():


    round_states = {
        "garden": generate_round_state("garden")[0],
        "cross": generate_round_state("cross")[0],
        "default": generate_round_state("default")[0]
    }
    env = GMSEnv({"round_states": round_states, "max_episode_steps": 128})
    env = ActionWrapper(env)          # same wrapper as PPO

    # --- ADD THESE DEBUG PRINTS HERE ---
    print(f"\nDEBUG: Before check_env - Type of env: {type(env)}")
    if hasattr(env, 'reset'):
        print(f"DEBUG: Before check_env - env.reset signature: {inspect.signature(env.reset)}")
        print(f"DEBUG: Before check_env - Is env an instance of gym.Env? {isinstance(env, gym.Env)}")
    else:
        print("DEBUG: Before check_env - env object does not have a reset method!")
    # --- END DEBUG PRINTS ---

    check_env(env)
    # Monitor wrapper is crucial for getting episode rewards/lengths into logs
    return Monitor(env)

if __name__ == '__main__':
    CONTINUE_TRAINING = False
    MODEL_PATH = "models/a0e11luo/model.zip"
    
    # Extract run ID from model path for wandb resume
    run_id = None
    if CONTINUE_TRAINING and os.path.exists(MODEL_PATH):
        # Extract run ID from path like "models/v3ahj95o10model.zip"
        run_id = MODEL_PATH.split("/")[1]
        print(f"Continuing training from model: {MODEL_PATH}")
        print(f"Will resume wandb run: {run_id}")
    
    # Initialize W&B run (resume if continuing training)
    if run_id:
        run = wandb.init(
            project="my-game-rl-training",
            id=run_id,                    # Use existing run ID
            resume="must",                # Must resume (will fail if run doesn't exist)
            config=wandb_config,
            sync_tensorboard=True,
            save_code=True
        )
    else:
        run = wandb.init(
            project="my-game-rl-training",
            config=wandb_config,
            sync_tensorboard=True,
            save_code=True
        )

    env = SubprocVecEnv([make_env for _ in range(N_ENVS)])

    ckpt = torch.load("models/bc_policy.pt", map_location="cpu")  # Not .zip!

# --- PPO Agent Instantiation ---
    if CONTINUE_TRAINING and os.path.exists(MODEL_PATH):
        print(f"Loaded model from {MODEL_PATH} to continue training.")
        time.sleep(3)
        model = PPO.load(
            MODEL_PATH,
            env=env,
            custom_objects={"policy_class": DictPolicy},
            device="mps"
        )
    else:
        print("Started training from scratch.")
        time.sleep(3)
        model = PPO(
            DictPolicy,
            env,
            learning_rate=wandb_config["learning_rate"],
            n_steps=wandb_config["n_steps"],
            batch_size=wandb_config["batch_size"],
            n_epochs=wandb_config["n_epochs"],
            gamma=wandb_config["gamma"],
            gae_lambda=wandb_config["gae_lambda"],
            clip_range=wandb_config["clip_range"],
            ent_coef=wandb_config["ent_coef"],
            verbose=1,
            tensorboard_log=f"runs/{run.id}",
            device="mps"
        )
        model.policy.load_state_dict(ckpt["policy"])
    
    policy2 = DictPolicy(env.observation_space, env.action_space, lr_schedule=lambda x: 0.0003)
    policy_only = torch.load("models/bc_policy.pt", map_location="cpu")
    policy2.load_state_dict(policy_only["policy"])

    SAVE_FREQ = 4000

# --- WandbCallback Configuration ---
# This callback explicitly manages saving models and logging gradients.
    wandb_callback = WandbCallback(
        model_save_path=f"models/{run.id}", # Path to save models within the W&B run directory
        model_save_freq=SAVE_FREQ // N_ENVS, # Convert to steps per environment if n_steps is per env
                                             # The callback expects total timesteps after an update (n_steps * N_ENVS)
                                             # but if you want it every 100k total steps, it's simpler to set based on total.
                                             # (Or just use SAVE_FREQ as 100_000 if that's your total step interval)
        gradient_save_freq=1000, # Log gradient histograms every 1000 steps
        verbose=2,
    )
    model.policy.eval()
    test_env = make_env()
    obs, _ = test_env.reset()
    action = model.predict(obs, deterministic=True)
    print("action: ", action)
    action2 = policy2.predict(obs, deterministic=True)
    print("action2: ", action2)
    print(policy2.state_dict().keys())
    print(policy2.state_dict()["policy.fc1.weight"])
    print(policy2.state_dict()["policy.fc1.bias"])
    time.sleep(1000000)

# --- Train the Agent ---
    print("Starting training with W&B logging...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[wandb_callback, TimingCallback()],
    )
    print("Training finished.")

# Save the final model directly to W&B (optional, as model_save_freq handles it)
# wandb.save(os.path.join(f"models/{run.id}", "final_model.zip"))

# Finish the W&B run
    run.finish()

# Close the environment
    env.close()
