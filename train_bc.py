"""
train_bc.py — Approach 1: Behavioral Cloning (IL only).

Loads human demonstrations from the RoboCasa dataset, trains a state-based
MLP policy via supervised learning (MSE loss on actions), then evaluates it
and saves all artefacts to results/.

Usage:
    python train_bc.py [options]

Outputs (all inside --output, default results/):
    bc_model.pt          — full checkpoint: state_dict + metadata
    bc_losses.json       — per-epoch average MSE loss
    bc_eval.json         — quick evaluation metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import robocasa  # noqa: F401 — registers envs
import robocasa.macros as _macros
import robocasa.utils.dataset_registry as _reg
import robocasa.utils.lerobot_utils as _LU

from common import BCPolicy, RESULTS_DIR
from gym_wrapper import KettleLidGymEnv, make_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ENV_NAME = "OpenElectricKettleLid"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

class Transition(NamedTuple):
    obs: np.ndarray      # flat float32 observation
    action: np.ndarray   # flat float32 action


def _default_dataset_base_path() -> Path:
    """Return the dataset base directory used by RoboCasa's registry."""
    if _macros.DATASET_BASE_PATH is not None:
        return Path(_macros.DATASET_BASE_PATH).expanduser()
    return Path(robocasa.__path__[0]).parent.absolute() / "datasets"


def _bundled_modality_path() -> Path:
    """Return RoboCasa's built-in PandaOmron modality spec."""
    return (
        Path(robocasa.__path__[0])
        / "models"
        / "assets"
        / "groot_dataset_assets"
        / "PandaOmron_modality.json"
    )


def _looks_like_lerobot_dataset(path: Path) -> bool:
    """Heuristic check for a LeRobot dataset root."""
    return path.is_dir() and (path / "data").is_dir() and (path / "extras").is_dir()


def _registry_relative_path(registered_path: Path) -> Path:
    """Recover the portable dataset subpath from an absolute registry path."""
    try:
        return registered_path.relative_to(_default_dataset_base_path())
    except ValueError:
        parts = registered_path.parts
        for marker in ("v1.0", "real_data"):
            if marker in parts:
                return Path(*parts[parts.index(marker):])
        return Path(registered_path.name)


def _resolve_dataset_path(
    *,
    split: str,
    source: str,
    dataset_root: Path | None = None,
) -> Path:
    """
    Resolve the LeRobot dataset path from the registry plus optional overrides.

    `dataset_root` may be either:
      - the RoboCasa datasets base directory containing `v1.0/`
      - the task directory containing `lerobot/`
      - the `lerobot/` dataset directory itself
    """
    meta = _reg.get_ds_meta(ENV_NAME, split=split, source=source, demo_fraction=1.0)
    if meta is None:
        raise RuntimeError(
            f"Dataset not found for {ENV_NAME!r} split={split!r} source={source!r}.\n"
            "Download with:\n"
            "  python -m robocasa.scripts.download_datasets --tasks OpenElectricKettleLid"
        )

    registered_path = Path(meta["path"]).expanduser()

    override = dataset_root
    if override is None:
        env_override = os.environ.get("ROBOCASA_DATASET_BASE_PATH")
        if env_override:
            override = Path(env_override).expanduser()

    if override is None:
        return registered_path

    if _looks_like_lerobot_dataset(override):
        return override

    if _looks_like_lerobot_dataset(override / "lerobot"):
        return override / "lerobot"

    return override / _registry_relative_path(registered_path)


def _load_modality_config(ds_path: Path) -> dict:
    """Load dataset modality metadata, falling back to RoboCasa's bundled spec."""
    modality_path = ds_path / "meta" / "modality.json"
    if modality_path.is_file():
        return json.loads(modality_path.read_text())

    fallback_path = _bundled_modality_path()
    if fallback_path.is_file():
        logger.warning(
            "Dataset is missing %s; falling back to bundled modality metadata at %s.",
            modality_path,
            fallback_path,
        )
        return json.loads(fallback_path.read_text())

    raise RuntimeError(
        f"Dataset is missing {modality_path} and no bundled fallback was found at "
        f"{fallback_path}."
    )


def _raise_missing_dataset_error(ds_path: Path, *, split: str, source: str) -> None:
    """Raise an actionable error for missing or incomplete local datasets."""
    raise RuntimeError(
        f"Dataset files for {ENV_NAME!r} split={split!r} source={source!r} "
        "are not installed locally.\n"
        f"Expected a LeRobot dataset at:\n  {ds_path}\n"
        "The dataset directory must contain at least `data/` and `extras/`.\n"
        "Fix this by either:\n"
        "  1. Downloading the RoboCasa dataset:\n"
        "     python -m robocasa.scripts.download_datasets --tasks OpenElectricKettleLid\n"
        "  2. Pointing this script at an existing dataset location:\n"
        "     python train_bc.py --dataset-root /path/to/robocasa/datasets\n"
        "     python train_bc.py --dataset-root /path/to/.../OpenElectricKettleLid/20250820\n"
        "     python train_bc.py --dataset-root /path/to/.../OpenElectricKettleLid/20250820/lerobot\n"
        "  3. Setting `DATASET_BASE_PATH` in `robocasa/macros_private.py`\n"
        "  4. Exporting `ROBOCASA_DATASET_BASE_PATH=/path/to/robocasa/datasets`"
    )


def load_transitions(
    env: KettleLidGymEnv,
    *,
    n_episodes: int,
    split: str = "pretrain",
    source: str = "human",
    dataset_root: Path | None = None,
) -> list[Transition]:
    """
    Replay human demonstrations and collect (obs, action) pairs.

    The environment is reset to the exact initial state of each demo
    episode, then replayed action-by-action to collect state observations
    that are consistent with the gym wrapper's flattening logic.

    Args:
        env:         An instantiated KettleLidGymEnv used for stepping.
        n_episodes:  Maximum number of episodes to load.
        split:       RoboCasa dataset split ("pretrain" | "target").
        source:      Demonstration source ("human").
        dataset_root: Optional dataset override. May point to the RoboCasa
                      datasets base directory, the task folder, or the
                      LeRobot dataset directory itself.

    Returns:
        List of (obs, action) Transition objects.

    Raises:
        RuntimeError: If the dataset is not found locally.
    """
    ds_path = _resolve_dataset_path(split=split, source=source, dataset_root=dataset_root)
    if not _looks_like_lerobot_dataset(ds_path):
        _raise_missing_dataset_error(ds_path, split=split, source=source)

    modality_cfg = _load_modality_config(ds_path)
    episode_paths = _LU.get_episodes(ds_path)
    n_load = min(n_episodes, len(episode_paths))

    transitions: list[Transition] = []
    inner = env._env  # underlying KettleLidEnv (robosuite)

    for ep_idx in range(n_load):
        logger.info("Loading episode %d / %d …", ep_idx + 1, n_load)

        # ── Reset env to demo initial state ──────────────────────────
        ep_meta = _LU.get_episode_meta(ds_path, ep_idx)
        initial_state = _LU.get_episode_states(ds_path, ep_idx)[0]
        model_xml = _LU.get_episode_model_xml(ds_path, ep_idx)

        inner.set_ep_meta(ep_meta)
        inner.reset()
        xml = inner.edit_model_xml(model_xml)
        inner.reset_from_xml_string(xml)
        inner.sim.reset()
        inner.sim.set_state_from_flattened(initial_state)
        inner.sim.forward()

        # ── Build flat action from modality config ────────────────────
        parquet_files = list(ds_path.glob(f"data/*/episode_{ep_idx:06d}.parquet"))
        if not parquet_files:
            logger.warning("No parquet file for episode %d, skipping.", ep_idx)
            inner.unset_ep_meta()
            continue

        df = pd.read_parquet(parquet_files[0])

        # Get first observation after the reset
        raw_obs = inner._get_observations()
        current_obs = env._flatten(raw_obs)

        for _, row in df.iterrows():
            action_parts = []
            for _name, info in modality_cfg["action"].items():
                key = info["original_key"]
                segment = np.array(row[key][info["start"]: info["end"]])
                action_parts.append(segment)
            flat_action = np.concatenate(action_parts).astype(np.float32)

            transitions.append(Transition(obs=current_obs.copy(), action=flat_action))

            # Step forward
            raw_next, _r, done, _info = inner.step(flat_action.astype(np.float64))
            current_obs = env._flatten(raw_next)
            if done:
                break

        inner.unset_ep_meta()

    logger.info(
        "Loaded %d transitions from %d / %d episodes.",
        len(transitions),
        n_load,
        len(episode_paths),
    )
    return transitions


def transitions_to_tensors(
    transitions: list[Transition],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack transitions into (obs_tensor, action_tensor)."""
    obs = torch.from_numpy(np.stack([t.obs for t in transitions]))
    act = torch.from_numpy(np.stack([t.action for t in transitions]))
    return obs, act


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(
    obs: torch.Tensor,
    actions: torch.Tensor,
    policy: BCPolicy,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> list[float]:
    """
    Train BCPolicy with MSE loss.

    Returns a list of per-epoch average losses.
    """
    policy = policy.to(device)
    obs = obs.to(device)
    actions = actions.to(device)

    loader = DataLoader(
        TensorDataset(obs, actions),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.MSELoss()

    epoch_losses: list[float] = []
    for epoch in range(epochs):
        policy.train()
        batch_total = 0.0
        for obs_b, act_b in loader:
            pred = policy(obs_b)
            loss = criterion(pred, act_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_total += loss.item()

        avg = batch_total / len(loader)
        epoch_losses.append(avg)

        if (epoch + 1) % 10 == 0:
            logger.info("  epoch %3d / %d   loss = %.6f", epoch + 1, epochs, avg)

    return epoch_losses


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    policy: BCPolicy,
    env: KettleLidGymEnv,
    *,
    n_episodes: int,
    device: torch.device,
) -> dict:
    """
    Roll out BCPolicy deterministically for n_episodes.

    Returns a dict with: success_rate, avg_lid_progress, avg_reward,
    avg_episode_length.
    """
    policy = policy.to(device).eval()
    successes = 0
    lid_progs: list[float] = []
    rewards: list[float] = []
    lengths: list[int] = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0

        while not done:
            action = policy.predict_numpy(obs, device=device)
            obs, r, terminated, truncated, info = env.step(action)
            ep_reward += r
            ep_len += 1
            done = terminated or truncated

        successes += int(info.get("success", False))
        lid_progs.append(float(info.get("lid_progress", 0.0)))
        rewards.append(ep_reward)
        lengths.append(ep_len)

    return {
        "success_rate": successes / n_episodes,
        "avg_lid_progress": float(np.mean(lid_progs)),
        "std_lid_progress": float(np.std(lid_progs)),
        "avg_reward": float(np.mean(rewards)),
        "avg_episode_length": float(np.mean(lengths)),
        "n_episodes": n_episodes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    policy: BCPolicy,
    losses: list[float],
    output_dir: Path,
) -> Path:
    """
    Save policy weights + metadata into a single .pt file.

    Also writes a human-readable losses.json sidecar.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "bc_model.pt"

    torch.save(
        {
            "state_dict": policy.state_dict(),
            "obs_dim": policy.obs_dim,
            "act_dim": policy.act_dim,
            "hidden_dim": policy.hidden_dim,
            "losses": losses,
        },
        model_path,
    )

    (output_dir / "bc_losses.json").write_text(
        json.dumps({"losses": losses}, indent=2)
    )
    logger.info("Checkpoint saved → %s", model_path)
    return model_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Approach 1: Behavioral Cloning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--episodes",   type=int,   default=50,          help="Demo episodes to load")
    p.add_argument("--epochs",     type=int,   default=100,         help="Training epochs")
    p.add_argument("--batch-size", type=int,   default=256,         help="Mini-batch size")
    p.add_argument("--lr",         type=float, default=3e-4,        help="Adam learning rate")
    p.add_argument("--hidden-dim", type=int,   default=256,         help="MLP hidden size")
    p.add_argument("--eval-ep",    type=int,   default=20,          help="Eval episodes after training")
    p.add_argument("--output",     type=Path,  default=RESULTS_DIR, help="Output directory")
    p.add_argument("--split",      type=str,   default="pretrain",  help="Dataset split")
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Optional dataset override. Can point to the RoboCasa datasets base "
            "directory, the task folder, or the LeRobot dataset directory itself"
        ),
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Environment ─────────────────────────────────────────────────
    logger.info("Initialising environment …")
    env = make_env(reward_mode="dense", seed=0)

    # ── Data ────────────────────────────────────────────────────────
    transitions = load_transitions(
        env,
        n_episodes=args.episodes,
        split=args.split,
        dataset_root=args.dataset_root,
    )
    if not transitions:
        logger.error("No transitions loaded. Aborting.")
        sys.exit(1)

    obs_t, act_t = transitions_to_tensors(transitions)
    obs_dim, act_dim = obs_t.shape[1], act_t.shape[1]
    logger.info(
        "Dataset: obs_dim=%d  act_dim=%d  n_transitions=%d",
        obs_dim, act_dim, len(transitions),
    )

    # ── Training ────────────────────────────────────────────────────
    logger.info("Training BC policy …")
    policy = BCPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim)
    losses = train(
        obs_t, act_t, policy,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )
    logger.info("Final loss: %.6f", losses[-1])

    # ── Save ────────────────────────────────────────────────────────
    save_checkpoint(policy, losses, args.output)

    # ── Evaluate ────────────────────────────────────────────────────
    logger.info("Evaluating BC policy (%d episodes) …", args.eval_ep)
    metrics = evaluate(policy, env, n_episodes=args.eval_ep, device=device)
    (args.output / "bc_eval.json").write_text(json.dumps(metrics, indent=2))
    logger.info(
        "BC eval: success_rate=%.1f%%  avg_lid=%.3f",
        metrics["success_rate"] * 100,
        metrics["avg_lid_progress"],
    )

    env.close()


if __name__ == "__main__":
    main()
