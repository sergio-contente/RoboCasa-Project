from collections import deque
from typing import Generic, SupportsFloat, TypeVar
import random

import numpy as np


Observation = TypeVar("Observation")
Action = TypeVar("Action")

class ReplayBuffer(Generic[Observation, Action]):
    """
    A class used to save and replay data.
    """
    buffer: deque[tuple[Observation, Action, SupportsFloat, Observation, bool]]

    def __init__(self, buffer_size: int = 10_000):
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

    def random_batch(self, batch_size: int):
        """
        Return a batch of size `batch_size`.
        """
        observations, actions, rewards, next_observations, dones = zip(*random.sample(self.buffer, batch_size))
        return np.stack(observations), actions, rewards, np.stack(next_observations), dones

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

    from environment_transformer import ActionObservationTransformer
    

    env = ActionObservationTransformer(
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

    assert observations.shape[0] == BATCH_SIZE
    assert next_observations.shape[0] == BATCH_SIZE
    assert len(actions) == BATCH_SIZE
    assert len(rewards) == BATCH_SIZE
    assert len(dones) == BATCH_SIZE
