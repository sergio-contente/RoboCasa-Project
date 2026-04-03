import math
from model.embedding import ObservationEmbedding
from abc import abstractmethod
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

from environment_transformer import ActionObservationTransformer, Observation
from model.actor import ActorRNN
from model.critic import CriticRNN
from replay_buffer import ReplayBuffer
import utils

@dataclass
class ContinuousAction:
    action_prob_mean: torch.Tensor
    action_prob_var: torch.Tensor

    @staticmethod
    def concat(values: list["ContinuousAction"]) -> "ContinuousAction":
        return ContinuousAction(
            action_prob_mean = utils.concat_tensors([
                value.action_prob_mean for value in values
            ]),
            action_prob_var = utils.concat_tensors([
                value.action_prob_var for value in values
            ])
        )


class ActorCriticAgent:
    environment: ActionObservationTransformer
    actor: ActorRNN
    actor_optim: torch.optim.Adam
    
    def __init__(self, env: ActionObservationTransformer):
        assert env.action_space.shape is not None

        self.environment = env

        (action_space,) = env.action_space.shape
        #pyrefly: ignore 
        video_observation_space_nb_channels: int = env.observation_space["video"].shape[-1]
        #pyrefly: ignore 
        (other_observation_space_shape, ) = env.observation_space["other"].shape

        self.actor = ActorRNN(
            action_space,
            video_observation_space_nb_channels,
            other_observation_space_shape,

            observation_embedding_size=32,
            action_embedding_size=32,
            reward_embedding_size=16,

            rnn_class = nn.LSTM,
            rnn_hidden_size = 64,
            rnn_num_layers = 1,
        ).to(utils.device)
        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(),
            lr=1e-4
        )

    def learn(self, nb_episodes: int):
        env = self.environment
        
        for episode_id in range(nb_episodes):
            print(f"=== Episode {episode_id} ===")
            observation, _ = env.reset()
            self.learn_episode(observation)
            print()
    
    @abstractmethod
    def learn_episode(self, initial_observation: Observation):
        ...

class OneStepACAgent(ActorCriticAgent):
    buffer: ReplayBuffer
    gamma: float

    critic: ObservationEmbedding
    critic_optim: torch.optim.Adam

    def __init__(self, env: ActionObservationTransformer):
        super().__init__(env)
        self.buffer = ReplayBuffer()
        self.gamma = .3

        #pyrefly: ignore 
        video_observation_space_nb_channels: int = env.observation_space["video"].shape[-1]
        #pyrefly: ignore 
        (other_observation_space_shape, ) = env.observation_space["other"].shape

        self.critic = ObservationEmbedding(
            video_observation_space_nb_channels=video_observation_space_nb_channels,
            other_observation_space_shape=other_observation_space_shape,
            output_size=1,
        ).to(utils.device)
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(),
            lr=1e-4
        )

    def learn_episode(self, initial_observation: Observation):
        env = self.environment

        observation = initial_observation
        action: torch.Tensor = utils.get_tensor(
            #pyrefly: ignore unsupported-operation
            env.action_space.sample() * 0.
        )

        done = False
        reward = utils.scalar_to_tensor(0.)
        step=0
        while not done and step < 2000:
            step+=1
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()

            previous_observation=Observation(
                video=observation.video.detach(),
                other=observation.other.detach()
            )
            previous_action=action.detach()
            
            # Pick an action
            action_mean, action_var = self.actor(
                observation=previous_observation,
                previous_action=previous_action
            )
            action = torch.normal(action_mean[0], torch.sqrt(action_var[0]))
            observation, reward, terminated, truncated, _info = env.step(
                action.detach().cpu().numpy()
            )
            done = terminated or truncated

            # Compute the loss
            #observation: Observation, reward: SupportsFloat, previous_action: torch.Tensor, action: torch.Tensor) -> list[torch.Tensor]:
            prev_value = self.critic(previous_observation)[0]
            value = self.critic(observation)[0] 
            delta = float(reward + self.gamma * value - prev_value)
            
            loss_critic: torch.Tensor = -delta*value
            loss_critic.backward()
            self.critic_optim.step()
            
            loss_actor: torch.Tensor = (
                (action - action_mean[0])**2 / (2.*action_var[0]) +
                (2.*math.pi*action_var[0]).log()
            ).sum()
            loss_actor *= delta
            loss_actor.backward()
            self.actor_optim.step()

            print(f"Step: {step}: Loss actor: {float(loss_actor)}; Loss critic: {float(loss_critic)}")


class SACAgent(ActorCriticAgent):
    buffer: ReplayBuffer
    critic: CriticRNN
    critic_optim: torch.optim.Adam

    def __init__(self, env: ActionObservationTransformer):
        super().__init__(env)
        self.buffer = ReplayBuffer()

        #pyrefly: ignore 
        (action_space,) = env.action_space.shape
        #pyrefly: ignore 
        video_observation_space_nb_channels: int = env.observation_space["video"].shape[-1]
        #pyrefly: ignore 
        (other_observation_space_shape, ) = env.observation_space["other"].shape

        self.critic = CriticRNN(
            action_space,
            video_observation_space_nb_channels,
            other_observation_space_shape,

            observation_embedding_size=32,
            action_embedding_size=32,
            reward_embedding_size=16,

            rnn_class = nn.LSTM,
            rnn_hidden_size = 64,
            rnn_num_layers = 2,

            nb_q_estimations=1
        ).to(utils.device)
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(),
            lr=1e-4
        )
    
    def learn_episode(self, initial_observation: Observation):
        #
        # TODO
        #
        env = self.environment

        observation = initial_observation
        action: torch.Tensor = utils.get_tensor(
            #pyrefly: ignore unsupported-operation
            env.action_space.sample() * 0.
        )

        done = False
        reward = utils.scalar_to_tensor(0.)
        step=0
        while not done:
            print(f"Step: {step}", end="\r"); step+=1

            with torch.no_grad():
                previous_observation=observation
                previous_action=action
                action_mean, action_var = self.actor(
                    observation=previous_observation,
                    previous_action=previous_action
                )
                action = torch.normal(action_mean[0], torch.sqrt(action_var[0]))
                observation, reward, terminated, truncated, _info = env.step(
                    action.detach().cpu().numpy()
                )
                done = terminated or truncated
                self.buffer.add_sample(
                    previous_observation,
                    previous_action,
                    ContinuousAction(
                        action_prob_mean=action_mean,
                        action_prob_var=action_var
                    ),
                    reward,
                    observation,
                    done
                )

            if step % 200 == 0:
                observations, prev_actions,  actions, rewards, next_observations, dones = self.buffer.random_batch(128)
                x = self.actor(
                    observation=previous_observation,
                    previous_action=previous_action
                )
                loss: torch.Tensor = (x[0]**2 + x[1]**2).sum()
                loss.backward()
                self.actor_optim.step()
