"""
train_gail.py — Approach 1 (GAIL): Generative Adversarial Imitation Learning.

Uses the `imitation` library's GAIL to train an adversarial imitator on human
demonstrations for the OpenElectricKettleLid task.

Unlike plain Behavioral Cloning (MSE on actions), GAIL trains jointly:
  - A *generator* (PPO policy) that tries to produce transitions the
    discriminator cannot distinguish from expert demonstrations.
  - A *discriminator* that classifies (obs, act) pairs as expert or learner.

The reward signal used for PPO updates comes entirely from the discriminator,
not from the environment reward — making GAIL robust to distribution shift.

Dependencies (install before running):
    pip install imitation==1.0.1 stable-baselines3==2.2.1

Usage:
    python train_gail.py [options]

Outputs (all inside --output, default results/):
    gail_model.zip       — trained PPO generator (SB3 format)
    gail_eval.json       — evaluation metrics after training
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.types import Trajectory
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm

import robocasa  # noqa: F401 — registers envs
import robocasa.utils.dataset_registry as _reg
import robocasa.utils.lerobot_utils as _LU

from common import RESULTS_DIR
from gym_wrapper import KettleLidGymEnv, make_env
from train_bc import (
    _load_modality_config,
    _looks_like_lerobot_dataset,
    _raise_missing_dataset_error,
    _resolve_dataset_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ENV_NAME = "OpenElectricKettleLid"


# ─────────────────────────────────────────────────────────────────────────────
# Expert trajectory loading
# ─────────────────────────────────────────────────────────────────────────────

def load_expert_trajectories(
    env: KettleLidGymEnv,
    *,
    n_episodes: int,
    split: str = "pretrain",
    source: str = "human",
    dataset_root: Path | None = None,
) -> list[Trajectory]:
    """
    Replay human demonstrations and package them as `imitation` Trajectory objects.

    Each trajectory contains:
      obs  : float32 array of shape (T+1, obs_dim) — obs at every step + terminal
      acts : float32 array of shape (T, act_dim)   — actions taken
      infos: None
      terminal: True (each demo episode ends naturally)

    Raises:
        RuntimeError: if the dataset is not found locally.
    """
    ds_path = _resolve_dataset_path(split=split, source=source, dataset_root=dataset_root)
    if not _looks_like_lerobot_dataset(ds_path):
        _raise_missing_dataset_error(ds_path, split=split, source=source)

    modality_cfg = _load_modality_config(ds_path)
    episode_paths = _LU.get_episodes(ds_path)
    n_load = min(n_episodes, len(episode_paths))

    trajectories: list[Trajectory] = []
    inner = env._env  # underlying KettleLidEnv (robosuite)

    for ep_idx in range(n_load):
        logger.info("Loading episode %d / %d …", ep_idx + 1, n_load)

        ep_meta      = _LU.get_episode_meta(ds_path, ep_idx)
        initial_state = _LU.get_episode_states(ds_path, ep_idx)[0]
        model_xml    = _LU.get_episode_model_xml(ds_path, ep_idx)

        # ── Reset to demo initial state ──────────────────────────────────
        inner.set_ep_meta(ep_meta)
        inner.reset()
        xml = inner.edit_model_xml(model_xml)
        inner.reset_from_xml_string(xml)
        inner.sim.reset()
        inner.sim.set_state_from_flattened(initial_state)
        inner.sim.forward()

        parquet_files = list(ds_path.glob(f"data/*/episode_{ep_idx:06d}.parquet"))
        if not parquet_files:
            logger.warning("No parquet file for episode %d, skipping.", ep_idx)
            inner.unset_ep_meta()
            continue

        df = pd.read_parquet(parquet_files[0])

        obs_list: list[np.ndarray] = []
        act_list: list[np.ndarray] = []

        raw_obs = inner._get_observations()
        obs_list.append(env._flatten(raw_obs))

        for _, row in df.iterrows():
            action_parts = []
            for _name, info in modality_cfg["action"].items():
                key     = info["original_key"]
                segment = np.array(row[key][info["start"]: info["end"]])
                action_parts.append(segment)
            flat_action = np.concatenate(action_parts).astype(np.float32)
            act_list.append(flat_action)

            raw_next, _r, done, _info = inner.step(flat_action.astype(np.float64))
            obs_list.append(env._flatten(raw_next))
            if done:
                break

        inner.unset_ep_meta()

        if not act_list:
            logger.warning("Episode %d had no actions, skipping.", ep_idx)
            continue

        trajectories.append(
            Trajectory(
                obs=np.array(obs_list, dtype=np.float32),   # (T+1, obs_dim)
                acts=np.array(act_list, dtype=np.float32),  # (T, act_dim)
                infos=None,
                terminal=True,
            )
        )

    total_transitions = sum(len(t.acts) for t in trajectories)
    logger.info(
        "Loaded %d expert trajectories (%d total transitions).",
        len(trajectories), total_transitions,
    )
    return trajectories


# ─────────────────────────────────────────────────────────────────────────────
# GAIL setup
# ─────────────────────────────────────────────────────────────────────────────

def build_gail(
    venv: DummyVecEnv,
    expert_trajs: list[Trajectory],
    *,
    demo_batch_size: int,
    n_disc_updates: int,
    n_steps: int,
    batch_size: int,
    learning_rate: float,
    log_dir: str,
) -> tuple[GAIL, PPO]:
    """
    Construct the GAIL trainer and its PPO generator.

    The generator uses a two-layer [256, 256] MLP matching the architecture
    used in other approaches so that pre-trained BC weights can be transferred.

    Returns:
        (gail_trainer, learner) — both are needed to train and save.
    """
    learner = PPO(
        "MlpPolicy",
        venv,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
    )

    reward_net = BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
    )

    gail_trainer = GAIL(
        demonstrations=expert_trajs,
        demo_batch_size=demo_batch_size,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        n_disc_updates_per_round=n_disc_updates,
        log_dir=log_dir,
        allow_variable_horizon=True,  # episodes can end early on success
    )

    return gail_trainer, learner


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    learner: PPO,
    *,
    n_episodes: int,
    seed: int = 42,
) -> dict:
    """
    Roll out the trained GAIL generator for n_episodes and return metrics.

    Evaluation uses the true environment reward (dense mode, threshold 0.95)
    — not the discriminator reward — so results are comparable across approaches.
    """
    env = make_env(reward_mode="dense", seed=seed)

    successes = 0
    lid_progs: list[float] = []
    rewards: list[float] = []
    lengths: list[int] = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0
        info: dict = {}

        while not done:
            action, _ = learner.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            ep_reward += float(r)
            ep_len += 1
            done = terminated or truncated

        successes += int(info.get("success", False))
        lid_progs.append(float(info.get("lid_progress", 0.0)))
        rewards.append(ep_reward)
        lengths.append(ep_len)

    env.close()

    return {
        "success_rate":       successes / n_episodes,
        "avg_lid_progress":   float(np.mean(lid_progs)),
        "std_lid_progress":   float(np.std(lid_progs)),
        "avg_reward":         float(np.mean(rewards)),
        "avg_episode_length": float(np.mean(lengths)),
        "n_episodes":         n_episodes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Approach 1 (GAIL): Generative Adversarial Imitation Learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--episodes",       type=int,   default=50,          help="Expert demo episodes to load")
    p.add_argument("--timesteps",      type=int,   default=500_000,     help="Total GAIL training timesteps")
    p.add_argument("--n-envs",         type=int,   default=4,           help="Parallel generator environments")
    p.add_argument("--lr",             type=float, default=3e-4,        help="PPO generator learning rate")
    p.add_argument("--n-steps",        type=int,   default=2048,        help="PPO n_steps per env per update")
    p.add_argument("--batch-size",     type=int,   default=256,         help="PPO mini-batch size")
    p.add_argument("--demo-batch",     type=int,   default=1024,        help="Expert transitions per disc update")
    p.add_argument("--n-disc-updates", type=int,   default=4,           help="Discriminator updates per round")
    p.add_argument("--eval-ep",        type=int,   default=20,          help="Evaluation episodes after training")
    p.add_argument("--split",          type=str,   default="pretrain",  help="Dataset split")
    p.add_argument("--output",         type=Path,  default=RESULTS_DIR, help="Output directory")
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional dataset override (datasets base dir, task folder, or lerobot dir)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Environment (for data loading + generation) ──────────────────
    logger.info("Initialising environment …")
    env_single = make_env(reward_mode="dense", seed=0)

    # ── Expert demonstrations ────────────────────────────────────────
    logger.info("Loading %d expert episodes …", args.episodes)
    expert_trajs = load_expert_trajectories(
        env_single,
        n_episodes=args.episodes,
        split=args.split,
        dataset_root=args.dataset_root,
    )
    env_single.close()

    if not expert_trajs:
        raise RuntimeError("No expert trajectories loaded — aborting.")

    # ── Vectorised training environments ────────────────────────────
    logger.info("Building %d parallel environments …", args.n_envs)
    venv = DummyVecEnv([
        (lambda i: lambda: make_env(reward_mode="dense", seed=i))(i)
        for i in range(args.n_envs)
    ])

    # ── GAIL trainer ─────────────────────────────────────────────────
    log_dir = str(output_dir / "gail_logs")
    os.makedirs(log_dir, exist_ok=True)

    logger.info(
        "Building GAIL (demo_batch=%d, n_disc_updates=%d) …",
        args.demo_batch, args.n_disc_updates,
    )
    gail_trainer, learner = build_gail(
        venv,
        expert_trajs,
        demo_batch_size=args.demo_batch,
        n_disc_updates=args.n_disc_updates,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        log_dir=log_dir,
    )

    # ── Training ─────────────────────────────────────────────────────
    logger.info("Training GAIL for %d timesteps …", args.timesteps)
    gail_trainer.train(total_timesteps=args.timesteps)

    # ── Save ─────────────────────────────────────────────────────────
    model_path = str(output_dir / "gail_model")
    learner.save(model_path)
    logger.info("Generator saved → %s.zip", model_path)

    # ── Evaluate ─────────────────────────────────────────────────────
    logger.info("Evaluating GAIL generator (%d episodes) …", args.eval_ep)
    metrics = evaluate(learner, n_episodes=args.eval_ep)
    (output_dir / "gail_eval.json").write_text(json.dumps(metrics, indent=2))
    logger.info(
        "GAIL eval: success_rate=%.1f%%  avg_lid=%.3f",
        metrics["success_rate"] * 100,
        metrics["avg_lid_progress"],
    )

    venv.close()


if __name__ == "__main__":
    main()
