import os

import gymnasium as gym
import imageio
import numpy as np
from tqdm import tqdm

from dataset_manager import DatasetManager, reset_based_on_episode
from replay_buffer import ReplayBuffer
from environment_transformer import ActionObservationTransformer, Observation

def load_dataset(env_name: str):
    buffer: ReplayBuffer[Observation, np.ndarray] = ReplayBuffer()

    print(f"Loading: {env_name}")

    env = ActionObservationTransformer(
        gym.make(
            f"robocasa/{env_name}",
            split="pretrain", # use 'pretrain' or 'target' kitchen scenes and objects
            seed=0 # seed environment as needed. set seed=None to run unseeded
        ),
        [ "annotation.human.task_description" ]
    )
    dataset = DatasetManager(
        env_name,
        split="pretrain",
        source="human",
    )

    for ind in range(5):
        print(f"Loading episode {ind+1} / {5}")
        ep_metadata, initial_state_flatten, model, actions = dataset.get_episode_actions(ind)
        reset_based_on_episode(env, ep_metadata, model, initial_state_flatten)

        observation, _ = env.reset()
        for action in tqdm(actions):
            action = env.reverse_action(action)
            next_observation, reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated
            buffer.add_sample(observation, action, reward, next_observation, done)
            if done:
                break
            observation = next_observation
        env.get_wrapper_attr("unset_ep_meta")()
    
    return env, buffer

VIDEO_PATH = "./test.mp4"
CAMERA_NAME = "video.robot0_eye_in_hand"
env, dataset = load_dataset("OpenElectricKettleLid")

os.makedirs(os.path.dirname(VIDEO_PATH), exist_ok=True)
with imageio.get_writer(VIDEO_PATH, fps=20) as video_handler:
    print(f"=== Saving the steps in {VIDEO_PATH} ===")

    for (obs, _, _, _, _) in tqdm(dataset.buffer):
        proper_obs = env.reverse_observation(obs)
        video_handler.append_data(proper_obs[CAMERA_NAME])
