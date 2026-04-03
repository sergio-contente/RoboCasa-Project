from collections import deque
from typing import Generic, TypeVar
import random

import numpy as np
import numpy.typing as npt
import torch

from environment_transformer import Observation
import utils

Action = TypeVar("Action")
class ReplayBuffer(Generic[Action]):
    """
    A class used to save and replay data.
    """
    buffer: deque[tuple[Observation, Action, Action, torch.Tensor, Observation, bool]]

    def __init__(self, buffer_size: int = 2_500):
        self.buffer = deque(maxlen=buffer_size)

    def add_sample(
        self,
        observation: Observation,
        prev_action: Action,
        action: Action,
        reward: torch.Tensor,
        next_observation: Observation,
        done: bool
    ):
        """
        Add a transition tuple.
        """
        self.buffer.append((observation, prev_action, action, reward, next_observation, done))

    def _concat_actions(self, actions: list[Action]) -> Action:
        action_type = type(actions[0])
        if action_type is np.ndarray:
            actions_concatenated = np.concat([
                np.expand_dims(act, axis=0)
                for act in actions
            ], axis=0)
            return actions_concatenated
        elif hasattr(action_type, "concat"):
            return  action_type.concat(actions)
        else:
            raise ValueError("We don't know how to concatenate this")


    def random_batch(self, batch_size: int) -> tuple[Observation, Action, Action, torch.Tensor, Observation, npt.NDArray[np.bool]]:
        """
        Return a batch of size `batch_size`.
        """
        assert batch_size > 0
        samples: list[tuple[Observation, Action, Action, torch.Tensor, Observation, bool]] = random.sample(self.buffer, batch_size)
        
        return (
            # Initial observations
            Observation(
                video=utils.concat_tensors([
                    sample[0].video
                    for sample in samples
                ]),
                other=utils.concat_tensors([
                    sample[0].other
                    for sample in samples
                ])
            ),
            # Previous actions
            self._concat_actions([
                sample[1] for sample in samples
            ]),
            # Actions
            self._concat_actions([
                sample[2] for sample in samples
            ]),
            # Rewards
            utils.concat_tensors([
                sample[3]
                for sample in samples
            ]),
            # next observations
            Observation(
                video=utils.concat_tensors([
                    sample[4].video
                    for sample in samples
                ]),
                other=utils.concat_tensors([
                    sample[4].other
                    for sample in samples
                ]),
            ),
            # is done?
            np.array([
                float(sample[5])
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

    env.reset()
    observation = None
    action = None
    for _ in tqdm(range(NB_STEPS)):
        prev_observation = observation
        prev_action = action

        action = env.action_space.sample()
        assert isinstance(action, np.ndarray)

        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if prev_observation is not None and prev_action is not None:
            buffer.add_sample(prev_observation, prev_action, action, reward, observation, done)

    BATCH_SIZE = 10
    (
        observations,
        prev_actions,
        actions,
        rewards,
        next_observations,
        dones
    ) = buffer.random_batch(BATCH_SIZE)

    assert len(prev_actions) == BATCH_SIZE
    assert len(actions) == BATCH_SIZE
    assert rewards.shape == (BATCH_SIZE, 1)
    assert len(dones) == BATCH_SIZE
    assert isinstance(observations, environment_transformer.Observation)
    assert isinstance(next_observations, environment_transformer.Observation)
    assert observations.video.shape[0] == BATCH_SIZE
    assert observations.other.shape[0] == BATCH_SIZE
    assert next_observations.video.shape[0] == BATCH_SIZE
    assert next_observations.other.shape[0] == BATCH_SIZE
