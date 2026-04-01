import numpy as np
import torch
import torch.nn as nn

from environment_transformer import Observation

class DenseEmbedding(nn.Module):
    def __init__(self,
        input_size: int,
        output_size: int,

        activation: type[nn.Module] = nn.ReLU
    ):
        super(DenseEmbedding, self).__init__()
        self.input_size  = input_size
        self.output_size = output_size

        # Based on Li et al.'s idea
        self.layer = nn.Linear(
            self.input_size,
            self.output_size,
            bias=True
        )
        self.activation = activation()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        x = self.layer(x)
        x = self.activation(x)
        return x

class ImageEmbedding(nn.Module):
    def __init__(self,
        input_size: tuple[int, int, int],
        nb_fully_conencted_layers: int,
        dim_hidden_layer: int,
        output_size: int
    ):
        super(ImageEmbedding, self).__init__()
        self.input_size  = input_size
        self.output_size = output_size

        layers = []

        # Build the convolutional layers
        #TODO

        # Build the final dense layers
        layers.append(nn.Flatten())
        for i in range(nb_fully_conencted_layers):
            layers.append(
                DenseEmbedding(
                    input_size=... if i == 0 else dim_hidden_layer,
                    output_size=dim_hidden_layer if i < nb_fully_conencted_layers-1 else output_size,
                    activation=nn.ReLU
                )
            )
        self.layers = nn.Sequential(layers)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)

class ObservationEmbedding(nn.Module):
    def __init__(self,
        video_observation_space_shape: tuple[int, int, int],
        other_observation_space_shape: int,
        output_size: int,
    ):
        super(ObservationEmbedding, self).__init__()
        self.video_embedding = ImageEmbedding(
            video_observation_space_shape,
            output_size
        )
        self.other_embedding = DenseEmbedding(
            other_observation_space_shape,
            output_size
        )
    
    def forward(self, observation: Observation) -> torch.Tensor:
        return (
            self.video_embedding(observation.video) +
            self.other_embedding(observation.other)
        )