import os
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.data.types import Trajectory
from imitation.util.logger import configure
import robocasa
import robosuite
from dataset_manager import load_dataset
from environment_transformer import ActionObservationTransformer
import utils


class SB3DictWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        video_space = env.observation_space.spaces["video"]
        other_space = env.observation_space.spaces["other"]

        video_shape = (
            video_space.shape[2],
            video_space.shape[0],
            video_space.shape[1],
        )
        total_dim = int(np.prod(video_shape)) + int(other_space.shape[0])

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32
        )

    def observation(self, obs):
        video_flat = obs.video.cpu().numpy().transpose(2, 0, 1).flatten()
        other_flat = obs.other.cpu().numpy().flatten()
        return np.concatenate([video_flat, other_flat]).astype(np.float32)


class ExtractPretrainedModel(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        self.video_shape = (9, 256, 256)
        self.video_dim = int(np.prod(self.video_shape))
        other_dim = observation_space.shape[0] - self.video_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(self.video_shape[0], 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        cnn_out_dim = 128 * 4 * 4

        self.mlp = nn.Sequential(
            nn.Linear(cnn_out_dim + other_dim, features_dim), nn.ReLU()
        )

        model_path = "chunked_imitation_model.pth"

        if os.path.exists(model_path):
            model_weights = torch.load(
                model_path, map_location="cpu", weights_only=True
            )
            cnn_weights = {
                k.replace("cnn.", ""): v for k, v in model_weights.items() if "cnn" in k
            }
            self.cnn.load_state_dict(cnn_weights)
            for param in self.cnn.parameters():
                param.requires_grad = False

    def forward(self, observations) -> torch.Tensor:
        video_flat = observations[:, : self.video_dim]
        other = observations[:, self.video_dim :]

        video = video_flat.view(-1, *self.video_shape)
        video = self.cnn(video)
        merged = torch.cat([video, other], dim=-1)
        return self.mlp(merged)


def convert_to_trajectories(replay_buffer):
    trajectories = []
    obs_arrays = []
    acts = []

    for step in replay_buffer.buffer:
        obs, action, _, _, done = step

        obs_arrays.append(obs)
        acts.append(action)

        if done:
            obs_arrays.append(obs_arrays[-1])

            trajectory = Trajectory(
                obs=np.array(obs_arrays),
                acts=np.array(acts),
                infos=None,
                terminal=True,
            )
            trajectories.append(trajectory)

            obs_arrays = []
            acts = []

    if len(acts) > 0:
        obs_arrays.append(obs_arrays[-1])
        trajectory = Trajectory(
            obs=np.array(obs_arrays),
            acts=np.array(acts),
            infos=None,
            terminal=True,
        )
        trajectories.append(trajectory)

    return trajectories


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.set_device(device)

    def make_env():
        env = ActionObservationTransformer(
            gym.make(
                "robocasa/OpenElectricKettleLid",
                split="pretrain",
                seed=42,
            ),
            observation_spaces_to_discard=["annotation.human.task_description"],
        )
        env = SB3DictWrapper(env)
        return env

    template_env = make_env()

    train_replay_buffer = load_dataset(template_env, "OpenElectricKettleLid", 50)
    expert_trajectories = convert_to_trajectories(train_replay_buffer)

    venv = DummyVecEnv([make_env])

    policy_kwargs = dict(
        features_extractor_class=ExtractPretrainedModel,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    learner = PPO(
        env=venv,
        policy="MultiInputPolicy",
        policy_kwargs=policy_kwargs,
        learning_rate=3e-5,
        n_steps=1024,
        batch_size=64,
        device=device,
    )

    reward_net = BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
    )

    custom_logger = configure(
        folder="gail_tensorboard_logs", format_strs=["tensorboard", "stdout"]
    )

    gail_trainer = GAIL(
        demonstrations=expert_trajectories,
        demo_batch_size=64,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=4,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
        custom_logger=custom_logger,
    )

    gail_trainer.train(total_timesteps=100_000)

    learner.save("gail_model")


if __name__ == "__main__":
    main()
