"""
custom_env.py — KettleLidEnv

A clean subclass of OpenElectricKettleLid with:
  - configurable reward mode ("dense" | "sparse")
  - configurable and dynamically-updatable success threshold (for curriculum)
  - isolated, documented reward shaping
  - defensive helpers for eef position and kettle position

Dense reward:
    r = 0.3 * r_reach + 1.0 * r_lid + r_success
    where:
        r_reach   = 1 - tanh(5 * dist(eef, kettle))   ∈ (0, 1]
        r_lid     = lid_progress                        ∈ [0, 1]
        r_success = 10.0 if lid >= threshold else 0.0

Sparse reward:
    r = 10.0 if lid >= threshold else 0.0
"""

from __future__ import annotations

import logging

import numpy as np
from robocasa.environments.kitchen.atomic.kitchen_electric_kettle import (
    OpenElectricKettleLid,
)

logger = logging.getLogger(__name__)

REWARD_REACH_SCALE = 5.0   # tanh shaping scale for eef-to-kettle distance
REWARD_REACH_WEIGHT = 0.3
REWARD_LID_WEIGHT = 1.0
REWARD_SUCCESS_BONUS = 10.0


class KettleLidEnv(OpenElectricKettleLid):
    """
    OpenElectricKettleLid with configurable reward and curriculum support.

    Args:
        reward_mode:        "dense" or "sparse".
        success_threshold:  Lid-opening ratio that counts as success [0, 1].
        **kwargs:           Forwarded to OpenElectricKettleLid / Kitchen.
    """

    def __init__(
        self,
        *args,
        reward_mode: str = "dense",
        success_threshold: float = 0.95,
        **kwargs,
    ) -> None:
        if reward_mode not in ("dense", "sparse"):
            raise ValueError(
                f"reward_mode must be 'dense' or 'sparse', got {reward_mode!r}"
            )
        # Set fields before super().__init__ so any init-time callbacks can use them
        self._reward_mode = reward_mode
        self._success_threshold = float(success_threshold)
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def set_success_threshold(self, threshold: float) -> None:
        """
        Dynamically update the success threshold.

        Called by CurriculumCallback via VecEnv.env_method().
        """
        self._success_threshold = float(threshold)
        logger.debug("success_threshold → %.2f", self._success_threshold)

    def lid_progress(self) -> float:
        """Return lid opening ratio ∈ [0, 1].  0 = closed, 1 = fully open."""
        state = self.electric_kettle.get_state(self)
        return float(state.get("lid", 0.0))

    # ------------------------------------------------------------------ #
    # Overrides
    # ------------------------------------------------------------------ #

    def _check_success(self) -> bool:
        return self.lid_progress() >= self._success_threshold

    def reward(self, action: np.ndarray | None = None) -> float:  # type: ignore[override]
        lid = self.lid_progress()

        if self._reward_mode == "sparse":
            return REWARD_SUCCESS_BONUS if lid >= self._success_threshold else 0.0

        # ── Dense ──────────────────────────────────────────────────────
        eef = self._eef_pos()
        kettle = self._kettle_pos()
        dist = float(np.linalg.norm(eef - kettle))

        r_reach = 1.0 - np.tanh(REWARD_REACH_SCALE * dist)
        r_lid = lid
        r_success = REWARD_SUCCESS_BONUS if self._check_success() else 0.0

        return float(
            REWARD_REACH_WEIGHT * r_reach
            + REWARD_LID_WEIGHT * r_lid
            + r_success
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _eef_pos(self) -> np.ndarray:
        """
        Return right-arm end-effector position as a (3,) float64 array.

        Falls back to zeros if the site lookup fails (e.g., during init).
        """
        try:
            eef_ids = self.robots[0].eef_site_id
            if isinstance(eef_ids, dict):
                sid = eef_ids.get("right") or next(iter(eef_ids.values()))
            else:
                sid = eef_ids
            return np.array(self.sim.data.site_xpos[sid], dtype=np.float64)
        except Exception:
            return np.zeros(3, dtype=np.float64)

    def _kettle_pos(self) -> np.ndarray:
        """
        Return electric-kettle body position as a (3,) float64 array.

        Falls back to zeros if the body-id lookup fails.
        """
        try:
            body_id = self.obj_body_id[self.electric_kettle.name]
            return np.array(self.sim.data.body_xpos[body_id], dtype=np.float64)
        except Exception:
            return np.zeros(3, dtype=np.float64)
