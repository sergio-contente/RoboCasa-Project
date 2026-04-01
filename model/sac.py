from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from environment_transformer import ActionObservationTransformer, Observation
from model.actor import ActorRNN
from model.critic import CriticRNN
from replay_buffer import ReplayBuffer
import utils

@dataclass
class ContinuousAction:
    action_prob_mean: torch.Tensor
    action_prob_var: torch.Tensor

class SACAgent:
    buffer: ReplayBuffer[Observation, ContinuousAction]

    def __init__(self, env: ActionObservationTransformer):
        assert env.action_space.shape is not None

        self.environment = env
        self.buffer = ReplayBuffer()

        (action_space,) = env.action_space.shape
        #pyrefly: ignore 
        video_observation_space_shape: tuple[int, int, int] = env.observation_space["video"].shape
        #pyrefly: ignore 
        (other_observation_space_shape, ) = env.observation_space["other"].shape

        self.actor = ActorRNN(
            action_space,
            video_observation_space_shape,
            other_observation_space_shape,

            observation_embedding_size=32,
            action_embedding_size=32,
            reward_embedding_size=16,

            rnn_class = nn.LSTM,
            rnn_hidden_size = 64,
            rnn_num_layers = 1,
        ).to(utils.device)
        self.critic = CriticRNN(
            action_space,
            video_observation_space_shape,
            other_observation_space_shape,

            observation_embedding_size=32,
            action_embedding_size=32,
            reward_embedding_size=16,

            rnn_class = nn.LSTM,
            rnn_hidden_size = 64,
            rnn_num_layers = 2
        ).to(utils.device)

    def learn(self, nb_episodes: int):
        env = self.environment

        for episode_id in tqdm(range(nb_episodes)):
            observation, _ = env.reset()
            #pyrefly: ignore bad-assignment
            action: np.ndarray = env.action_space.sample()
            action *= 0.

            done = False
            reward = 0.
            while not done:
                previous_observation=observation
                previous_action=action
                action_mean, action_var = self.actor(
                    observation=previous_observation,
                    reward=reward,
                    previous_action=previous_action
                )
                action = torch.normal(action_mean, torch.sqrt(action_var)).detach().numpy()
                observation, reward, terminated, truncated, _info = env.step(action)
                done = terminated or truncated
                self.buffer.add_sample(
                    previous_observation,
                    ContinuousAction(
                        action_prob_mean=action_mean,
                        action_prob_var=action_var
                    ),
                    reward,
                    observation,
                    done
                )
