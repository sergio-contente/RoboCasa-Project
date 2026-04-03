import os

import gymnasium as gym
import robocasa
import torch

from environment_transformer import ActionObservationTransformer
from model.sac import OneStepACAgent
import utils

VIDEO_PATH = "./test.mp4"
CAMERA_NAME = "video.robot0_eye_in_hand"
ENV_NAME = "OpenElectricKettleLid"

os.makedirs(os.path.dirname(VIDEO_PATH), exist_ok=True)
print(f"Is CUDA available: {torch.cuda.is_available()}")
utils.set_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

print(f"Loading: {ENV_NAME}")
env = ActionObservationTransformer(
    gym.make(
        f"robocasa/{ENV_NAME}",
        split="pretrain", # use 'pretrain' or 'target' kitchen scenes and objects
        seed=0, # seed environment as needed. set seed=None to run unseeded
        renderer="mjviewer"
    ),
    [ "annotation.human.task_description" ]
)

sac = OneStepACAgent(env)
sac.learn(5)
