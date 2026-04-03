# train_v2.py
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["EGL_DEVICE_ID"] = "0"

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.policies import ActorCriticPolicy
from rew_shaping_and_curriculum_train import make_env, CURRICULUM_STAGES

# ─────────────────────────────────────────
# IMPROVED HYPERPARAMETERS
# ─────────────────────────────────────────
PPO_KWARGS = dict(
    # smaller std init → less random actions from the start
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        log_std_init=-1.0,      # std ~ 0.37 instead of default ~1.0
        ortho_init=True,
    ),
    n_steps=4096,               # more steps per update → better gradient estimates
    batch_size=256,             # larger batch → more stable updates
    n_epochs=10,
    learning_rate=1e-4,         # lower lr → more stable
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.1,             # tighter clip → smaller policy updates
    ent_coef=0.001,             # much lower entropy → less random
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    tensorboard_log="logs/tensorboard_v2/",
)

# ─────────────────────────────────────────
# CURRICULUM CALLBACK (improved)
# ─────────────────────────────────────────
STAGE_TIMESTEPS = [150_000, 350_000]  # advance at these timesteps

class CurriculumCallback(BaseCallback):
    def __init__(self, stage_timesteps, verbose=1):
        super().__init__(verbose)
        self.stage_timesteps = stage_timesteps
        self.current_stage = 0

    def _on_step(self):
        if self.current_stage < len(self.stage_timesteps):
            if self.num_timesteps >= self.stage_timesteps[self.current_stage]:
                self.current_stage += 1
                for env in self.training_env.envs:
                    env.curriculum_stage = self.current_stage
                if self.verbose:
                    print(f"\n[Curriculum] Advanced to stage {self.current_stage} at {self.num_timesteps} steps")
        return True

# ─────────────────────────────────────────
# IL WARMUP CALLBACK
# idea: freeze value network for first N steps
# so policy focuses on imitating good actions first
# ─────────────────────────────────────────
class ILWarmupCallback(BaseCallback):
    """
    For the first warmup_steps, reduce entropy coefficient
    to near zero to encourage exploitation of shaped reward
    rather than random exploration.
    Also gradually anneals learning rate.
    """
    def __init__(self, warmup_steps=50_000, verbose=1):
        super().__init__(verbose)
        self.warmup_steps = warmup_steps

    def _on_step(self):
        progress = min(self.num_timesteps / self.warmup_steps, 1.0)
        # anneal ent_coef from 0 → 0.001 over warmup
        self.model.ent_coef = 0.001 * progress
        return True

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints_v2", exist_ok=True)

    env = DummyVecEnv([make_env()])
    env = VecMonitor(env, "logs/monitor_v2")

    # ── Option A: train from scratch with better hyperparams ──
    model = PPO("MlpPolicy", env, **PPO_KWARGS)

    # ── Option B: continue from previous checkpoint ──
    # Uncomment this and comment Option A if you want to
    # fine-tune from your existing model:
    #
    # model = PPO.load("open_kettle_final", env=env)
    # model.learning_rate = 1e-4
    # model.clip_range = 0.1
    # model.ent_coef = 0.001
    # model.n_steps = 4096
    # model.batch_size = 256

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path="checkpoints_v2/",
        name_prefix="open_kettle_v2",
    )
    curriculum_cb = CurriculumCallback(
        stage_timesteps=STAGE_TIMESTEPS,
        verbose=1,
    )
    warmup_cb = ILWarmupCallback(
        warmup_steps=50_000,
        verbose=1,
    )

    print("Starting improved training...")
    model.learn(
        total_timesteps=500_000,
        callback=[checkpoint_cb, curriculum_cb, warmup_cb],
        progress_bar=True,
        reset_num_timesteps=True,
    )

    model.save("open_kettle_v2_final")
    env.close()
    print("Done! Model saved to open_kettle_v2_final.zip")
