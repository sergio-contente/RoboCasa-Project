# train.py
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["EGL_DEVICE_ID"] = "0"

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robocasa.environments.kitchen.atomic.kitchen_electric_kettle import (
    OpenElectricKettleLid,
)

# ─────────────────────────────────────────────
# CURRICULUM CONFIG
# ─────────────────────────────────────────────
# 3 stages of increasing difficulty
# Stage 0: fixed layout, no style randomization
# Stage 1: random layout, fixed style
# Stage 2: full randomization (default robocasa behavior)

CURRICULUM_STAGES = [
    {"layout_ids": [0], "style_ids": [0]},       # stage 0 - easiest
    {"layout_ids": None, "style_ids": [0]},       # stage 1 - random layout
    {"layout_ids": None, "style_ids": None},      # stage 2 - full random
]

# Advance stage every N timesteps
STAGE_TIMESTEPS = [100_000, 200_000, 500_000]

# ─────────────────────────────────────────────
# CURRICULUM CALLBACK
# ─────────────────────────────────────────────
class CurriculumCallback(BaseCallback):
    """Advances curriculum stage based on timesteps."""
    def __init__(self, env_fn_list, stage_timesteps, verbose=1):
        super().__init__(verbose)
        self.stage_timesteps = stage_timesteps
        self.current_stage = 0

    def _on_step(self):
        if self.current_stage < len(self.stage_timesteps) - 1:
            if self.num_timesteps >= self.stage_timesteps[self.current_stage]:
                self.current_stage += 1
                # Update env curriculum stage
                for env in self.training_env.envs:
                    env.env.curriculum_stage = self.current_stage
                if self.verbose:
                    print(f"\n[Curriculum] Advanced to stage {self.current_stage}")
        return True

# ─────────────────────────────────────────────
# CURRICULUM-AWARE ENV WRAPPER
# ─────────────────────────────────────────────
class CurriculumEnv(gym.Wrapper):
    """
    Wraps OpenElectricKettleLid to support curriculum:
    resets with increasing scene randomization per stage.
    """
    def __init__(self, env):
        super().__init__(env)
        self.curriculum_stage = 0

    def reset(self, **kwargs):
        stage_cfg = CURRICULUM_STAGES[self.curriculum_stage]
        # Pass curriculum params into reset if supported
        # RoboCasa reset accepts layout_id and style_id
        try:
            obs, info = self.env.reset(
                layout_id=stage_cfg["layout_ids"][0]
                    if stage_cfg["layout_ids"] is not None else None,
                style_id=stage_cfg["style_ids"][0]
                    if stage_cfg["style_ids"] is not None else None,
            )
        except Exception:
            obs, info = self.env.reset()
        return obs, info

# ─────────────────────────────────────────────
# ENV FACTORY
# ─────────────────────────────────────────────
def make_env():
    def _init():
        env = OpenElectricKettleLid(
            robots=["PandaOmron"],
            controller_configs=None,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=["robot0_agentview_center", "robot0_eye_in_hand"],
            camera_heights=128,
            camera_widths=128,
            reward_shaping=True,
            horizon=500,
            ignore_done=False,
        )
        env.reset()
        env = GymWrapper(env)
        env = CurriculumEnv(env)
        return env
    return _init

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Single env for now (parallelism can be added later)
    env = DummyVecEnv([make_env()])
    env = VecMonitor(env, "logs/monitor")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,       # small entropy bonus for exploration
        tensorboard_log="logs/tensorboard/",
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path="checkpoints/",
        name_prefix="open_kettle_ppo",
    )

    curriculum_cb = CurriculumCallback(
        env_fn_list=None,
        stage_timesteps=STAGE_TIMESTEPS,
        verbose=1,
    )

    print("Starting training...")
    model.learn(
        total_timesteps=500_000,
        callback=[checkpoint_cb, curriculum_cb],
        progress_bar=True,
    )

    model.save("open_kettle_final")
    env.close()
    print("Done! Model saved to open_kettle_final.zip")
