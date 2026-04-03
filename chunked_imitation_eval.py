import os
import torch
import gymnasium as gym
import imageio
import robocasa
from environment_transformer import ActionObservationTransformer
from model.behaviour_cloning import BehaviourCloning
import utils


def eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.set_device(device)

    NUM_EPISODES = 20
    CHUNK_SIZE = 10
    SAVE_DIR = "eval_videos"
    os.makedirs(SAVE_DIR, exist_ok=True)

    base_env = gym.make(
        "robocasa/OpenElectricKettleLid",
        split="pretrain",
        seed=42,
    )

    env = ActionObservationTransformer(
        base_env,
        observation_spaces_to_discard=["annotation.human.task_description"],
    )

    obs, info = env.reset()
    video_channels = obs.video.shape[-1]
    other_dim = obs.other.shape[0]
    action_dim = env.action_space.shape[0]

    model = BehaviourCloning(
        video_channels=video_channels,
        other_dim=other_dim,
        action_dim=action_dim,
        chunk_size=CHUNK_SIZE,
        action_min=torch.zeros(action_dim),
        action_range=torch.ones(action_dim),
        device=device,
    )

    model.load_state_dict(
        torch.load(
            "chunked_imitation_model.pth", map_location=device, weights_only=True
        )
    )
    model.eval()

    successes = 0

    for ep in range(NUM_EPISODES):
        obs, info = env.reset()
        done = False
        step = 0
        max_steps = 1000

        frames = []

        while not done and step < max_steps:
            action_chunk = model.predict(obs)

            for i in range(CHUNK_SIZE):
                action = action_chunk[i]

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step += 1
                try:
                    render_obs = base_env.unwrapped.render()
                    if render_obs is not None:
                        frames.append(render_obs)
                except Exception as e:
                    pass

                is_success = info.get("is_success", False) or reward > 0.5
                if is_success:
                    done = True
                    break

                if done or step >= max_steps:
                    break

        if is_success:
            successes += 1

        if frames:
            video_path = os.path.join(SAVE_DIR, f"episode_{ep + 1}.mp4")
            imageio.mimsave(video_path, frames, fps=20)

    print(
        f"Success Rate: {successes}/{NUM_EPISODES} ({(successes/NUM_EPISODES)*100:.1f}%)"
    )


if __name__ == "__main__":
    eval()
