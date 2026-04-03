import numpy as np
import torch
import torchvision
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
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        assert len(x.shape) == 2, "Please provide a batch of data, or at least one"
        
        x = self.layer(x)
        x = self.activation(x)
        return x

class ObservationEmbedding(nn.Module):
    def __init__(self,
        video_observation_space_nb_channels: int,
        other_observation_space_shape: int,
        output_size: int,
    ):
        super(ObservationEmbedding, self).__init__()
        """
        The video embedding is just an adaption of AlexNet
        """
        self.video_embedding = nn.Sequential(
            torchvision.transforms.Resize((224, 224)),
            nn.Conv2d(
                in_channels=video_observation_space_nb_channels,
                out_channels=96,
                kernel_size=11,
                stride=4,
                padding='valid'
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(
                in_channels=96,
                out_channels=256,
                kernel_size=5,
                stride=1,
                padding='same'
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(
                in_channels=256,
                out_channels=384,
                kernel_size=3,
                stride=1,
                padding='same'
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=384,
                out_channels=384,
                kernel_size=3,
                stride=1,
                padding='same'
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=384,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding='same'
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Flatten(),
            nn.Linear(5*5*256, 4096),
            nn.ReLU(), nn.Dropout(.5),
            nn.Linear(4096, 4096),
            nn.ReLU(), nn.Dropout(.5),
            nn.Linear(4096, output_size)
        )
        self.other_embedding = DenseEmbedding(
            other_observation_space_shape,
            output_size
        )
    
    def forward(self, observation: Observation) -> torch.Tensor:
        video_tensor = observation.video
        if len(video_tensor.shape) == 3:
            video_tensor = video_tensor.unsqueeze(0)
        assert len(video_tensor.shape) == 4, "Please provide a batch of colored images, or at least one"

        video_tensor = (
            video_tensor.permute((0, 3, 1, 2))
                .to(torch.float32)
                .div(255)
                .contiguous()
        )
        return (
            self.video_embedding(video_tensor) +
            self.other_embedding(observation.other)
        )