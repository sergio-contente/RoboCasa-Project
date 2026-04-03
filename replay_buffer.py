from collections import deque
from typing import SupportsFloat
import random

import numpy as np
import numpy.typing as npt

from environment_transformer import Observation

Action = np.ndarray
class ReplayBuffer:
    """
    A class used to save and replay data.
    """
    buffer: deque[tuple[Observation, Action, SupportsFloat, Observation, bool]]

    def __init__(self, buffer_size: int = 2_000):
        self.buffer = deque(maxlen=buffer_size)

    def add_sample(
        self,
        observation: Observation,
        action: Action,
        reward: SupportsFloat,
        next_observation: Observation,
        done: bool
    ):
        """
        Add a transition tuple.
        """
        self.buffer.append((observation, action, reward, next_observation, done))

    def random_batch(self, batch_size: int) -> tuple[Observation, npt.NDArray[np.float32], npt.NDArray[np.float32], Observation, npt.NDArray[np.bool]]:
        """
        Return a batch of size `batch_size`.
        """
        samples: list[tuple[Observation, Action, SupportsFloat, Observation, bool]] = random.sample(self.buffer, batch_size)
        return (
            # Initial observations
            Observation(
                video=torch.concat([
                    sample[0].video.unsqueeze(0)
                    for sample in samples
                ], dim=0),
                other=torch.concat([
                    sample[0].other.unsqueeze(0)
                    for sample in samples
                ], dim=0)
            ),
            # Action
            np.concat([
                np.expand_dims(sample[1], axis=0)
                for sample in samples
            ], axis=0),
            # Rewards
            np.array([
                sample[2]
                for sample in samples
            ]),
            # next observations
            Observation(
                video=torch.concat([
                    sample[3].video.unsqueeze(0)
                    for sample in samples
                ], dim=0),
                other=torch.concat([
                    sample[3].other.unsqueeze(0)
                    for sample in samples
                ], dim=0)
            ),
            # is done?
            np.array([
                sample[4]
                for sample in samples
            ])
        )

if __name__ == "__main__":
    """
    Test the replay buffer by:
    - Sending a certain amount of steps
    - Retrieving a small batch
    - Checking if it has the right amount of states
    """
    import gymnasium as gym
    import robocasa
    from tqdm import tqdm
    import torch
    
    import environment_transformer
    import utils

    utils.set_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    env = environment_transformer.ActionObservationTransformer(
        gym.make(
            "robocasa/PickPlaceCounterToCabinet",
            split="pretrain", # use 'pretrain' or 'target' kitchen scenes and objects
            seed=0 # seed environment as needed. set seed=None to run unseeded
        ),
        [ "annotation.human.task_description" ]
    )

    buffer = ReplayBuffer()
    NB_STEPS = 100

    obs, info = env.reset()
    for _ in tqdm(range(NB_STEPS)):
        action = env.action_space.sample()
        assert isinstance(action, np.ndarray)

        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        buffer.add_sample(obs, action, reward, next_observation, done)

    BATCH_SIZE = 10
    observations, actions, rewards, next_observations, dones = buffer.random_batch(BATCH_SIZE)

    assert len(actions) == BATCH_SIZE
    assert len(rewards) == BATCH_SIZE
    assert len(dones) == BATCH_SIZE
    assert isinstance(observations, environment_transformer.Observation)
    assert isinstance(next_observations, environment_transformer.Observation)
    assert observations.video.shape[0] == BATCH_SIZE
    assert observations.other.shape[0] == BATCH_SIZE
    assert next_observations.video.shape[0] == BATCH_SIZE
    assert next_observations.other.shape[0] == BATCH_SIZE
