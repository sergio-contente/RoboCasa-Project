import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision.transforms as T
import wandb


class BehaviourCloningDataset(Dataset):

    def __init__(self, replay_buffer, chunk_size=10):
        self.data = list(replay_buffer.buffer)
        self.chunk_size = chunk_size

        all_actions = torch.tensor(
            np.array([item[1] for item in self.data]), dtype=torch.float32
        )
        self.action_min = all_actions.min(dim=0)[0]
        self.action_max = all_actions.max(dim=0)[0]
        self.action_range = torch.clamp(self.action_max - self.action_min, min=1e-6)
        self.color_jitter = T.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
        )
        self.random_crop = T.RandomCrop(size=(256, 256), padding=8, padding_mode="edge")

    def __len__(self):
        return len(self.data) - self.chunk_size

    def __getitem__(self, index):
        obs, _, _, _, _ = self.data[index]

        video_tensor = obs.video.permute(2, 0, 1).float() / 255.0
        jittered_chunks = [
            self.color_jitter(chunk) for chunk in torch.split(video_tensor, 3, dim=0)
        ]
        video_tensor = torch.cat(jittered_chunks, dim=0)
        video_tensor = self.random_crop(video_tensor)

        other_tensor = obs.other.float()

        actions = []
        for i in range(self.chunk_size):
            _, act, _, _, _ = self.data[index + i]
            actions.append(act)

        action_tensor = torch.tensor(np.array(actions), dtype=torch.float32)
        action_tensor = (
            2.0 * (action_tensor - self.action_min) / self.action_range - 1.0
        )
        action_tensor = action_tensor.flatten()

        return video_tensor, other_tensor, action_tensor

    def get_action_bounds(self):
        return self.action_min, self.action_range


class BehaviourCloning(nn.Module):
    def __init__(
        self,
        video_channels,
        other_dim,
        action_dim,
        chunk_size,
        action_min,
        action_range,
        lr=1e-4,
        device="cpu",
    ):
        super().__init__()
        self.device = device
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        self.register_buffer("action_min", action_min.to(device))
        self.register_buffer("action_range", action_range.to(device))

        self.cnn = nn.Sequential(
            nn.Conv2d(video_channels, 32, kernel_size=3, stride=2, padding=1),
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
            nn.Linear(cnn_out_dim + other_dim, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim * chunk_size),
            nn.Tanh(),
        )

        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def forward(self, video, other):
        video = self.cnn(video)
        merged = torch.cat([video, other], dim=-1)
        return self.mlp(merged)

    def train_policy(
        self, train_loader, val_loader, epochs=500, patience=15, use_wandb=True
    ):
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Training Phase
            self.train()
            train_loss = 0.0

            for batch_video, batch_other, batch_action in train_loader:
                batch_video = batch_video.to(self.device)
                batch_other = batch_other.to(self.device)
                batch_action = batch_action.to(self.device)

                pred = self.forward(batch_video, batch_other)
                loss = self.loss_fn(pred, batch_action)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation Phase
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_video, batch_other, batch_action in val_loader:
                    batch_video = batch_video.to(self.device)
                    batch_other = batch_other.to(self.device)
                    batch_action = batch_action.to(self.device)

                    pred = self.forward(batch_video, batch_other)
                    val_loss += self.loss_fn(pred, batch_action).item()

            avg_val_loss = val_loss / len(val_loader)

            if use_wandb:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                    }
                )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.state_dict(), "chunked_imitation_model.pth")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                self.load_state_dict(
                    torch.load(
                        "chunked_imitation_model.pth",
                        map_location=self.device,
                        weights_only=True,
                    )
                )
                break

    def predict(self, observation):
        self.eval()
        with torch.no_grad():
            video_tensor = (
                observation.video.permute(2, 0, 1).unsqueeze(0).float().to(self.device)
                / 255.0
            )
            other_tensor = observation.other.unsqueeze(0).float().to(self.device)

            normalized = self.forward(video_tensor, other_tensor)

            normalized_chunk = normalized.view(self.chunk_size, self.action_dim)

            unnormalized_chunk = (
                normalized_chunk + 1.0
            ) / 2.0 * self.action_range + self.action_min

            return unnormalized_chunk.cpu().numpy()
