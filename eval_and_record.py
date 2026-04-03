# eval_and_record.py
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["EGL_DEVICE_ID"] = "0"
os.environ["NUMBA_DISABLE_JIT"] = "1"

import numpy as np
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rew_shaping_and_curriculum_train import make_env, CURRICULUM_STAGES

N_EVAL_EPISODES_PER_STAGE = 25
VIDEO_FIRST_N = 2        # always record first N per stage
CAMERA = "robot0_agentview_center"
VIDEO_W, VIDEO_H = 512, 512
MODEL_PATH = "checkpoints_v2/open_kettle_v2_470000_steps"

EVAL_STAGES = [
    (0, "easy"),
    (2, "hard"),
]

def get_raw_env(vec_env):
    return vec_env.envs[0].env.env

def evaluate_stage(model, env, stage_id, stage_name, n_episodes, video_dir):
    os.makedirs(video_dir, exist_ok=True)
    env.envs[0].curriculum_stage = stage_id

    success_count = 0
    rewards = []
    ep_lengths = []

    print(f"\n--- Stage {stage_id} ({stage_name}) ---")

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        ep_len = 0
        frames = []
        recording = ep < VIDEO_FIRST_N

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            ep_len += 1

            if recording:
                try:
                    raw_env = get_raw_env(env)
                    frame = raw_env.sim.render(
                        camera_name=CAMERA,
                        width=VIDEO_W,
                        height=VIDEO_H,
                        depth=False,
                    )
                    frames.append(frame[::-1])
                except Exception as e:
                    print(f"  Frame capture failed: {e}")
                    recording = False

        raw_env = get_raw_env(env)
        success = raw_env.electric_kettle.get_state(raw_env)["lid"] >= 0.95
        if success:
            success_count += 1
            if not recording:  # record successes even if past first N
                recording = True

        rewards.append(ep_reward)
        ep_lengths.append(ep_len)

        label = "SUCCESS" if success else "FAIL"
        if len(frames) > 0:
            video_path = f"{video_dir}/ep{ep+1:03d}_{label}_rew{ep_reward:.0f}.mp4"
            imageio.mimsave(video_path, frames, fps=25)
            print(f"  Ep {ep+1:02d}: reward={ep_reward:.2f}, len={ep_len}, {label} → video saved")
        else:
            print(f"  Ep {ep+1:02d}: reward={ep_reward:.2f}, len={ep_len}, {label}")

    success_rate = success_count / n_episodes * 100
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_len = np.mean(ep_lengths)

    return dict(
        stage_id=stage_id,
        stage_name=stage_name,
        success_rate=success_rate,
        success_count=success_count,
        n_episodes=n_episodes,
        mean_reward=mean_reward,
        std_reward=std_reward,
        mean_len=mean_len,
    )

def main():
    print("Loading v2 model...")
    env = DummyVecEnv([make_env()])
    model = PPO.load(MODEL_PATH, env=env)
    print("Model loaded.")

    all_results = []
    for stage_id, stage_name in EVAL_STAGES:
        result = evaluate_stage(
            model=model,
            env=env,
            stage_id=stage_id,
            stage_name=stage_name,
            n_episodes=N_EVAL_EPISODES_PER_STAGE,
            video_dir=f"eval_videos_v2/stage_{stage_id}_{stage_name}",
        )
        all_results.append(result)

    # ─────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────
    summary_lines = [
        "===========================================",
        "EVALUATION SUMMARY — V2 MODEL",
        "===========================================",
        f"Model : {MODEL_PATH}",
        "",
    ]
    for r in all_results:
        summary_lines += [
            f"Stage {r['stage_id']} ({r['stage_name']:>4s}):",
            f"  Success rate       : {r['success_count']}/{r['n_episodes']} = {r['success_rate']:.1f}%",
            f"  Mean reward        : {r['mean_reward']:.2f} ± {r['std_reward']:.2f}",
            f"  Mean episode length: {r['mean_len']:.1f} steps",
            "",
        ]
    summary_lines.append("===========================================")
    summary = "\n".join(summary_lines)

    print("\n" + summary)
    with open("eval_summary_v2.txt", "w") as f:
        f.write(summary)
    print("\nSummary saved to eval_summary_v2.txt")
    print("Videos saved to eval_videos_v2/")

if __name__ == "__main__":
    main()
