from typing import Any, SupportsFloat, TypeVar

import gymnasium as gym
import numpy as np


ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")
WrapperActType = TypeVar("WrapperActType")
WrapperObsType = TypeVar("WrapperObsType")

class ActionObservationWrapper(gym.Wrapper[WrapperObsType, WrapperActType, ObsType, ActType]):
    """Superclass of wrappers that can modify observations using :meth:`observation` and
    actions using :meth:`action` for :meth:`reset` and :meth:`step`.
    """

    def __init__(self, env: gym.Env[ObsType, ActType]):
        """Constructor for the observation and action wrapper."""
        gym.Wrapper.__init__(self, env)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`reset`, returning a modified observation using :meth:`self.observation`."""
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`step` using :meth:`self.observation` on the returned observations."""
        observation, reward, terminated, truncated, info = self.env.step(self.action(action))
        return self.observation(observation), reward, terminated, truncated, info
    
    def action(self, action: WrapperActType) -> ActType:
        """Returns a modified action before :meth:`env.step` is called.

        Args:
            action: The original :meth:`step` actions

        Returns:
            The modified actions
        """
        raise NotImplementedError
    
    def observation(self, observation: ObsType) -> WrapperObsType:
        """Returns a modified observation.

        Args:
            observation: The :attr:`env` observation

        Returns:
            The modified observation
        """
        raise NotImplementedError


class ActionObservationTransformer(ActionObservationWrapper[dict, np.ndarray, dict, dict]):
    """
    This is the actual environment transformer we are going to use.
    """
    def __init__(self, env: gym.Env, observation_spaces_to_discard: list[str]):
        """
        Constructor.

        :param env: Environment we are going to wrap
        :param observation_spaces_to_discard: List of all the observation we want to ignore.
        """
        assert isinstance(env.observation_space, gym.spaces.Dict)
        ActionObservationWrapper.__init__(self, env)

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

        video_spaces: list[gym.spaces.Box] = list(video_space.values())
        video_dtype = video_spaces[0].dtype
        assert video_dtype is not None
        for i in range(len(video_spaces)):
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

        self._observation_space = gym.spaces.Dict({
            "video": self._intermediary_observation_space_video,
            "other": gym.spaces.flatten_space(self._intermediary_observation_space_other)
        })

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

    def action(self, action):
        return gym.spaces.utils.unflatten(self.env.action_space, action)

    def observation(self, observation: dict):
        video_value, other_value = self._sort_spaces(observation)
        output = {
            "video": np.concatenate(list(video_value.values()), axis=-1),
            "other": gym.spaces.utils.flatten(
                self._intermediary_observation_space_other,
                other_value
            )
        }

        return output

if __name__ == "__main__":
    import robocasa
    
    env = ActionObservationTransformer(
        gym.make(
            "robocasa/PickPlaceCounterToCabinet",
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
