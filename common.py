"""
common.py — Shared utilities used across training scripts.

Provides:
  - BCPolicy        : MLP policy used by train_bc.py and all IL+RL scripts.
  - transfer_bc_to_ppo : Defensive BC → SB3 PPO weight transfer.
  - load_bc_checkpoint : Load BCPolicy from disk.
  - RESULTS_DIR / DATA_DIR : canonical output paths.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from stable_baselines3 import PPO

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.resolve()
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# BC Policy (shared between train_bc.py and all IL+RL scripts)
# ─────────────────────────────────────────────────────────────────────────────

class BCPolicy(nn.Module):
    """
    Two-hidden-layer MLP policy for behavioral cloning.

    Architecture:
        obs_dim → hidden_dim → hidden_dim → act_dim  (Tanh output)

    The Tanh ensures actions stay in [-1, 1], matching the robosuite
    action spec convention.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),  # layer index 0
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # layer index 2
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),   # layer index 4
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_numpy(
        self,
        obs: "np.ndarray",  # type: ignore[name-defined]  # noqa: F821
        device: torch.device | None = None,
    ) -> "np.ndarray":  # type: ignore[name-defined]  # noqa: F821
        """Convenience: numpy in, numpy out (no grad, deterministic)."""
        import numpy as np

        dev = device or next(self.parameters()).device
        t = torch.tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
        with torch.no_grad():
            return self(t).squeeze(0).cpu().numpy()


def load_bc_checkpoint(path: str | Path, device: torch.device | None = None) -> BCPolicy:
    """
    Reconstruct a BCPolicy from a saved checkpoint.

    The checkpoint is expected to be either:
      - a raw state_dict (legacy), in which case the meta JSON is read
        from the same directory with suffix ``_meta.json``, or
      - a dict with keys ``"state_dict"``, ``"obs_dim"``, ``"act_dim"``,
        ``"hidden_dim"`` (preferred format).
    """
    path = Path(path)
    dev = device or torch.device("cpu")

    raw = torch.load(path, map_location=dev, weights_only=True)

    if isinstance(raw, dict) and "state_dict" in raw:
        obs_dim = raw["obs_dim"]
        act_dim = raw["act_dim"]
        hidden_dim = raw.get("hidden_dim", 256)
        state_dict = raw["state_dict"]
    else:
        # Legacy: raw state dict — try to read sidecar meta JSON
        state_dict = raw
        meta_path = path.with_suffix(".json")
        if not meta_path.exists():
            # Try _meta.json variant
            meta_path = path.parent / (path.stem + "_meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            obs_dim = meta["obs_dim"]
            act_dim = meta["act_dim"]
            hidden_dim = meta.get("hidden_dim", 256)
        else:
            raise FileNotFoundError(
                f"Cannot infer BCPolicy dims: neither embedded meta in {path} "
                f"nor sidecar {meta_path} found."
            )

    policy = BCPolicy(obs_dim, act_dim, hidden_dim)
    policy.load_state_dict(state_dict)
    policy.to(dev)
    policy.eval()
    return policy


# ─────────────────────────────────────────────────────────────────────────────
# BC → PPO weight transfer
# ─────────────────────────────────────────────────────────────────────────────

def transfer_bc_to_ppo(ppo_model: PPO, bc_policy: BCPolicy) -> None:
    """
    Copy weights from a BCPolicy MLP into an SB3 PPO MlpPolicy.

    Mapping (SB3 MlpPolicy with net_arch=[256, 256]):
        BC net.0  (Linear obs→hidden)   → mlp_extractor.policy_net.0
        BC net.2  (Linear hidden→hidden)→ mlp_extractor.policy_net.2
        BC net.4  (Linear hidden→act)   → action_net (mean actions)

    The transfer is purely best-effort:
      - If shapes match, weights are copied in-place.
      - If shapes differ (e.g., different hidden_dim), a clear warning is
        emitted and the mismatched layer is skipped.
      - Never raises; the model always trains, just possibly without transfer.
    """
    bc_sd = bc_policy.state_dict()
    ppo_policy = ppo_model.policy

    mapping = [
        ("net.0.weight", "mlp_extractor.policy_net.0.weight"),
        ("net.0.bias",   "mlp_extractor.policy_net.0.bias"),
        ("net.2.weight", "mlp_extractor.policy_net.2.weight"),
        ("net.2.bias",   "mlp_extractor.policy_net.2.bias"),
        ("net.4.weight", "action_net.weight"),
        ("net.4.bias",   "action_net.bias"),
    ]

    ppo_sd = ppo_policy.state_dict()
    n_transferred = 0

    for bc_key, ppo_key in mapping:
        if bc_key not in bc_sd:
            warnings.warn(f"[BC→PPO] BC key '{bc_key}' not found, skipping.")
            continue
        if ppo_key not in ppo_sd:
            warnings.warn(f"[BC→PPO] PPO key '{ppo_key}' not found, skipping.")
            continue

        bc_tensor = bc_sd[bc_key]
        ppo_tensor = ppo_sd[ppo_key]

        if bc_tensor.shape != ppo_tensor.shape:
            warnings.warn(
                f"[BC→PPO] Shape mismatch for '{bc_key}' → '{ppo_key}': "
                f"BC={tuple(bc_tensor.shape)} vs PPO={tuple(ppo_tensor.shape)}. "
                "Skipping this layer."
            )
            continue

        # Copy in-place on the correct device
        ppo_sd[ppo_key].copy_(bc_tensor)
        n_transferred += 1

    ppo_policy.load_state_dict(ppo_sd)
    logger.info("[BC→PPO] Transferred %d / %d parameter tensors.", n_transferred, len(mapping))
