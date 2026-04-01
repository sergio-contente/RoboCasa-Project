import os

import gymnasium as gym
import robocasa
#import imageio
import torch
#from tqdm import tqdm

from environment_transformer import ActionObservationTransformer
from model.sac import SACAgent
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
        seed=0 # seed environment as needed. set seed=None to run unseeded
    ),
    [ "annotation.human.task_description" ]
)

sac = SACAgent(env)
sac.learn(5)
#with imageio.get_writer(VIDEO_PATH, fps=20) as video_handler:
#    print(f"=== Saving the steps in {VIDEO_PATH} ===")
#
#    for (obs, _, _, _, _) in tqdm(dataset.buffer):
#        proper_obs = env.reverse_observation(obs)
#        video_handler.append_data(proper_obs[CAMERA_NAME])
