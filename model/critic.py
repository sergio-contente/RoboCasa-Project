from typing import Type, SupportsFloat

import numpy as np
import torch
import torch.nn as nn

from environment_transformer import Observation
from model.embedding import DenseEmbedding, ObservationEmbedding

class CriticRNN(nn.Module):
    def __init__(self,
        action_shape: int,
        video_observation_space_shape: tuple[int, int, int],
        other_observation_space_shape: int,

        observation_embedding_size: int,
        action_embedding_size: int,
        reward_embedding_size: int,

        rnn_class: Type[nn.Module],
        rnn_hidden_size: int,
        rnn_num_layers: int,
    ):
        super(CriticRNN, self).__init__()
        self.observation_for_rnn = ObservationEmbedding(
            video_observation_space_shape,
            other_observation_space_shape,
            observation_embedding_size
        )
        self.previous_action_embedding = DenseEmbedding(
            action_shape,
            action_embedding_size
        )
        self.reward_embedding = DenseEmbedding(
            1,
            reward_embedding_size
        )

        self.observation_for_output = ObservationEmbedding(
            video_observation_space_shape,
            other_observation_space_shape,
            rnn_hidden_size
        )
        self.action_for_output = DenseEmbedding(
            action_shape,
            rnn_hidden_size
        )
        
        self.rnn = rnn_class(
            input_size = (
                observation_embedding_size + action_embedding_size + reward_embedding_size
            ),
            hidden_size = rnn_hidden_size,
            num_layers = rnn_num_layers,
            batch_first=True,
            bias=True,
        )
        self.q1_layer = nn.Linear(rnn_hidden_size, 1)
        self.q2_layer = nn.Linear(rnn_hidden_size, 1)

    def forward(self, observation: Observation, reward: SupportsFloat, previous_action: np.ndarray, action: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.concat([
            self.observation_for_rnn(observation),
            self.reward_embedding(reward),
            self.previous_action_embedding(previous_action)
        ], dim=-1)
        hidden, _ = self.rnn(x)
        x = self.observation_for_output(observation) + self.action_for_output(action) + hidden
        return (
            self.q1_layer(x),
            self.q2_layer(x)
        )
