from dataclasses import dataclass
from typing import Any, SupportsFloat

import torch
import gymnasium as gym
import numpy as np

import utils

@dataclass
class Observation:
    video: torch.Tensor
    other: torch.Tensor

class ActionObservationTransformer(gym.Wrapper[Observation, np.ndarray, dict, dict]):
    """Superclass of wrappers that can modify observations using :meth:`observation` and
    actions using :meth:`action` for :meth:`reset` and :meth:`step`.
    """
    video_spaces_name: list[str]
    
    def __init__(self, env: gym.Env[dict, dict], observation_spaces_to_discard: list[str]):
        """Constructor for the observation and action wrapper."""
        assert isinstance(env.observation_space, gym.spaces.Dict)
        super(ActionObservationTransformer, self).__init__(env)

        self.observation_spaces_to_discard = observation_spaces_to_discard
        new_action_space = gym.spaces.utils.flatten_space(env.action_space)
        assert isinstance(new_action_space, gym.spaces.Box)
        self._action_space = new_action_space
        self._reward_range = env.reward_range
        self._metadata = env.metadata

        # The observation space needs a special treatment, as it contains
        # large features (video feeds), and converting & flattening data
        # would be terrible for memory (it would take 2MiB per observation).
        # But if we keep it as-is, we can cut the memory usage by four
        video_space: dict[str, gym.spaces.Space]
        other_space: dict[str, gym.spaces.Space]
        video_space, other_space = self._sort_spaces(env.observation_space)

        self.video_spaces_name = list(video_space.keys())
        video_spaces: list[gym.spaces.Box] = list(video_space.values())
        video_dtype = video_spaces[0].dtype
        assert video_dtype is not None

        for i in range(len(video_spaces)):
            # Check that all the video feeds are the same
            assert isinstance(video_spaces[i], gym.spaces.Box)
            if i > 0:
                assert video_spaces[i].low.shape == video_spaces[i-1].low.shape
                assert video_spaces[i].low.dtype == video_spaces[i-1].low.dtype
        
        self._intermediary_observation_space_video = gym.spaces.Box(
            low   = np.concatenate([cam.low  for cam in video_spaces], axis=-1),
            high  = np.concatenate([cam.high for cam in video_spaces], axis=-1),
            # pyrefly: ignore bad-argument-type
            dtype = video_dtype
        )
        self._intermediary_observation_space_other = gym.spaces.Dict(other_space)

        # pyrefly: ignore bad-argument-type
        self._observation_space = gym.spaces.Dict({
            "video": self._intermediary_observation_space_video,
            "other": gym.spaces.flatten_space(self._intermediary_observation_space_other)
        })

    def observation_sample(self) -> Observation:
        #pyrefly: ignore bad-assignment
        sample: dict = self.observation_space.sample()
        return Observation(
            other=utils.get_tensor(sample["other"]),
            video=utils.get_tensor(sample["video"]),
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`reset`, returning a modified observation using :meth:`self.observation`."""
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def step(
        self, action: np.ndarray
    ) -> tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`step` using :meth:`self.observation` on the returned observations."""
        observation, reward, terminated, truncated, info = self.env.step(self.action(action))
        return self.observation(observation), reward, terminated, truncated, info
    
    def _sort_spaces(self, spaces):
        """
        Separate video observations from the rest, and filter out unwanted dimensions
        """
        video = {}
        other = {}
        
        for space_name, space in spaces.items():
            feature: str
            if space_name in self.observation_spaces_to_discard:
                continue
            if space_name.startswith("video."):
                video[space_name] = space
            else:
                other[space_name] = space
            
        return (video, other)

    def action(self, action: np.ndarray):
        return gym.spaces.utils.unflatten(self.env.action_space, action)

    def observation(self, observation: dict) -> Observation:
        video_value, other_value = self._sort_spaces(observation)

        return Observation(
            video=utils.get_tensor(
                np.concatenate(list(video_value.values()), axis=-1)
            ),
            other=utils.get_tensor(
                np.array(gym.spaces.utils.flatten(
                    self._intermediary_observation_space_other,
                    other_value
                ))
            )
        )
    
    def reverse_action(self, action: dict) -> np.ndarray:
        return np.array(
            gym.spaces.utils.flatten(self.env.action_space, action)
        )

    def reverse_observation(self, observation: Observation) -> dict[str, np.ndarray]:
        output = {}
        output.update(
            gym.spaces.utils.unflatten(
                self._intermediary_observation_space_other,
                observation.other.cpu()
            )
        )

        nb_channels_per_dimension = observation.video.shape[-1] // len(self.video_spaces_name)
        for i, video_dimension in enumerate(self.video_spaces_name):
            output[video_dimension] = (
                observation.video[..., nb_channels_per_dimension*i : nb_channels_per_dimension*(i+1)]
                    .cpu().numpy()
            )
        return output

if __name__ == "__main__":
    import robocasa
    utils.set_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    env = ActionObservationTransformer(
        gym.make(
            "robocasa/OpenElectricKettleLid",
            split="pretrain", # use 'pretrain' or 'target' kitchen scenes and objects
            seed=0 # seed environment as needed. set seed=None to run unseeded
        ),
        [ "annotation.human.task_description" ]
    )
    
    print(
f"""==========
Environment: {env}
Action space: {env.action_space}
Observation space: {env.observation_space}
==========""")

    print("Check the observation functions")
    observation_from_env: dict = env.env.observation_space.sample()
    observation_from_transformer: Observation = env.observation_sample()

    new_observation_from_transformer = env.observation(env.reverse_observation(observation_from_transformer))
    np.testing.assert_array_equal(
        new_observation_from_transformer.other.cpu().numpy(),
        observation_from_transformer.other.cpu().numpy()
    )
    np.testing.assert_array_equal(
        new_observation_from_transformer.video.cpu().numpy(),
        observation_from_transformer.video.cpu().numpy()
    )

    new_observation_from_env = env.reverse_observation(env.observation(observation_from_env))
    missing_keys = set(new_observation_from_env.keys()) ^ set(observation_from_env.keys())
    assert missing_keys == set(env.observation_spaces_to_discard), \
        f"Missing keys: {missing_keys}"

    for key in new_observation_from_env.keys():
        np.testing.assert_array_equal(
            new_observation_from_env[key],
            observation_from_env[key]
        )


    print("Check the action functions")
    action_from_env: dict = env.env.action_space.sample()
    action_from_transformer: np.ndarray = np.array(env.action_space.sample())

    new_action_from_transformer = env.reverse_action(env.action(action_from_transformer))
    np.testing.assert_array_equal(
        new_action_from_transformer,
        action_from_transformer
    )

    new_action_from_env = env.action(env.reverse_action(action_from_env))
    assert set(new_action_from_env.keys()) == set(action_from_env.keys()), \
        f"Missing keys: {set(new_action_from_env.keys()) ^ (set(action_from_env.keys()))}"
    
    for key in new_action_from_env.keys():
        np.testing.assert_array_equal(
            new_action_from_env[key],
            action_from_env[key]
        )
