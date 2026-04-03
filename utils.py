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

def concat_tensors(l: list[torch.Tensor]) -> torch.Tensor:
    return torch.concat([elt.unsqueeze(0) for elt in l], dim=0)