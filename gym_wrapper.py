"""
gym_wrapper.py — SB3-compatible Gymnasium wrapper for KettleLidEnv.

Design choices:
  - State-based observations only (no images) for efficiency.
  - Observation keys are discovered dynamically at construction time by
    probing a reset() and selecting all flat (ndim ≤ 1) numpy arrays.
    This keeps the wrapper resilient to minor obs-key changes across
    RoboCasa versions.
  - Action space: the native flat float32 array from robosuite.
  - set_success_threshold() is exposed for VecEnv.env_method() calls
    made by CurriculumCallback.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from robosuite.controllers import load_composite_controller_config

from custom_env import KettleLidEnv

logger = logging.getLogger(__name__)

# Robosuite action spec returns (lo, hi) already at float64;
# we convert to float32 for SB3 compatibility.
_DTYPE = np.float32


class KettleLidGymEnv(gym.Env):
    """
    Gymnasium wrapper around KettleLidEnv.

    Observation space : flat float32 vector of all proprioceptive / object
                        state observations (no images).
    Action space      : flat float32 vector of the robosuite action spec
                        (12-dim for PandaOmron).

    Args:
        reward_mode:        "dense" or "sparse" — passed to KettleLidEnv.
        success_threshold:  Initial success threshold ∈ [0, 1].
        seed:               Integer seed forwarded to robosuite.
        max_episode_steps:  Hard truncation limit per episode.
        render_mode:        Unused; kept for Gymnasium API compliance.
    """

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        reward_mode: str = "dense",
        success_threshold: float = 0.95,
        seed: int = 0,
        max_episode_steps: int = 500,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self._step_count: int = 0

        # ── Build underlying KettleLidEnv ─────────────────────────────
        ctrl_cfg = load_composite_controller_config(controller=None, robot="PandaOmron")
        self._env = KettleLidEnv(
            robots="PandaOmron",
            controller_configs=ctrl_cfg,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            use_object_obs=True,
            reward_mode=reward_mode,
            success_threshold=success_threshold,
            control_freq=20,
            seed=seed,
        )

        # ── Probe observation space ───────────────────────────────────
        raw_obs = self._env.reset()
        self._obs_keys: list[str] = _select_obs_keys(raw_obs)
        obs_dim = int(sum(np.asarray(raw_obs[k]).flatten().shape[0] for k in self._obs_keys))
        logger.debug(
            "obs_keys=%s  obs_dim=%d",
            self._obs_keys,
            obs_dim,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=_DTYPE,
        )

        # ── Action space ──────────────────────────────────────────────
        a_lo, a_hi = self._env.action_spec
        self.action_space = spaces.Box(
            low=a_lo.astype(_DTYPE),
            high=a_hi.astype(_DTYPE),
            dtype=_DTYPE,
        )

    # ------------------------------------------------------------------ #
    # Gymnasium interface
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._env.seed = seed
        raw_obs = self._env.reset()
        self._step_count = 0
        info: dict = {
            "success": False,
            "lid_progress": self._env.lid_progress(),
        }
        return self._flatten(raw_obs), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        raw_obs, reward, done, _robosuite_info = self._env.step(
            action.astype(np.float64)
        )
        self._step_count += 1

        success = self._env._check_success()
        lid = self._env.lid_progress()
        terminated = bool(done) or success
        truncated = self._step_count >= self.max_episode_steps

        info: dict = {"success": success, "lid_progress": lid}
        return self._flatten(raw_obs), float(reward), terminated, truncated, info

    def render(self) -> None:
        return None

    def close(self) -> None:
        self._env.close()

    # ------------------------------------------------------------------ #
    # Curriculum support
    # ------------------------------------------------------------------ #

    def set_success_threshold(self, threshold: float) -> None:
        """Proxy forwarded via VecEnv.env_method() by CurriculumCallback."""
        self._env.set_success_threshold(threshold)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _flatten(self, raw_obs: dict) -> np.ndarray:
        """Flatten selected observation keys into a single float32 vector."""
        parts = [
            np.asarray(raw_obs[k]).flatten().astype(_DTYPE)
            for k in self._obs_keys
        ]
        return np.concatenate(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _select_obs_keys(raw_obs: dict) -> list[str]:
    """
    Return sorted list of keys whose values are flat (ndim ≤ 1) numpy arrays.

    This excludes image observations (shape (H, W, C)) and string annotations
    automatically, making the wrapper robust to obs-key changes.
    """
    return sorted(
        k
        for k, v in raw_obs.items()
        if isinstance(v, np.ndarray) and v.ndim <= 1
    )


def make_env(
    reward_mode: str = "dense",
    seed: int = 0,
    success_threshold: float = 0.95,
    max_episode_steps: int = 500,
) -> KettleLidGymEnv:
    """
    Factory that creates a single KettleLidGymEnv.

    Intended for use inside DummyVecEnv lambdas:
        DummyVecEnv([lambda i=i: make_env(seed=i) for i in range(4)])
    """
    return KettleLidGymEnv(
        reward_mode=reward_mode,
        seed=seed,
        success_threshold=success_threshold,
        max_episode_steps=max_episode_steps,
    )
