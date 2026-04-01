from dataset_manager import load_dataset
import os

import gymnasium as gym
import imageio
import numpy as np
from tqdm import tqdm

from dataset_manager import DatasetManager, reset_based_on_episode
from replay_buffer import ReplayBuffer
from environment_transformer import ActionObservationTransformer, Observation

VIDEO_PATH = "./test.mp4"
CAMERA_NAME = "video.robot0_eye_in_hand"
ENV_NAME = "OpenElectricKettleLid"

print(f"Loading: {ENV_NAME}")
env = ActionObservationTransformer(
    gym.make(
        f"robocasa/{ENV_NAME}",
        split="pretrain", # use 'pretrain' or 'target' kitchen scenes and objects
        seed=0 # seed environment as needed. set seed=None to run unseeded
    ),
    [ "annotation.human.task_description" ]
)

dataset = load_dataset(env, ENV_NAME, 5)

os.makedirs(os.path.dirname(VIDEO_PATH), exist_ok=True)
with imageio.get_writer(VIDEO_PATH, fps=20) as video_handler:
    print(f"=== Saving the steps in {VIDEO_PATH} ===")

    for (obs, _, _, _, _) in tqdm(dataset.buffer):
        proper_obs = env.reverse_observation(obs)
        video_handler.append_data(proper_obs[CAMERA_NAME])
