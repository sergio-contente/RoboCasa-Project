import numpy as np
import torch

device: torch.device | None = None
def set_device(dev: torch.device):
    global device
    device = dev

def get_tensor(array: np.ndarray) -> torch.Tensor:
    global device
    assert device is not None
    return torch.from_numpy(array).to(device=device)
