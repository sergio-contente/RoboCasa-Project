import gymnasium as gym
import robocasa
import torch
from tqdm import tqdm

from replay_buffer import ReplayBuffer
from environment_transformer import ActionObservationTransformer

env = ActionObservationTransformer(
    gym.make(
        "robocasa/CloseCabinet",
        split="pretrain", # use 'pretrain' or 'target' kitchen scenes and objects
        seed=0 # seed environment as needed. set seed=None to run unseeded
    ),
    [ "annotation.human.task_description" ]
)

print(f"""==========
Is CUDA available: {torch.cuda.is_available()}

Environment: {env}
==========""")

buffer = ReplayBuffer()
