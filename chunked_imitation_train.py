import torch
import wandb
from torch.utils.data import DataLoader, random_split
import gymnasium as gym

from environment_transformer import ActionObservationTransformer
from dataset_manager import load_dataset
from model.behaviour_cloning import BehaviourCloningDataset, BehaviourCloning
import utils


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.set_device(device)

    config = {
        "learning_rate": 1e-4,
        "batch_size": 32,
        "epochs": 500,
        "patience": 15,
        "episodes_loaded": 50,
        "chunk_size": 10,
    }

    wandb.init(project="robocasa", config=config)

    env = ActionObservationTransformer(
        gym.make("robocasa/OpenElectricKettleLid", split="pretrain", seed=None),
        observation_spaces_to_discard=["annotation.human.task_description"],
    )

    replay_buffer = load_dataset(
        env, "OpenElectricKettleLid", config["episodes_loaded"]
    )

    dataset = BehaviourCloningDataset(replay_buffer)
    action_min, action_range = dataset.get_action_bounds()

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    video, other, action = dataset[0]
    model = BehaviourCloning(
        video_channels=video.shape[0],
        other_dim=other.shape[0],
        action_dim=action.shape[0] // config["chunk_size"],
        action_min=action_min,
        action_range=action_range,
        chunk_size=config["chunk_size"],
        lr=config["learning_rate"],
        device=device,
    )

    model.train_policy(
        train_loader,
        val_loader,
        epochs=config["epochs"],
        patience=config["patience"],
        use_wandb=True,
    )

    wandb.finish()


if __name__ == "__main__":
    train()
