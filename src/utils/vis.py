# src/utils/vis.py

import torch
from PIL import Image


def denorm(x):
    mean = torch.tensor([0.5, 0.5, 0.5], device=x.device, dtype=x.dtype)
    std = torch.tensor([0.5, 0.5, 0.5], device=x.device, dtype=x.dtype)

    if x.dim() ==3:
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)

    elif x.dim() == 4:
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    
    else:
        raise ValueError("xi")

    return (x * std + mean).clamp(0.0, 1.0)

def to_pil(x):
    if x.dim() != 3:
        raise ValueError("must be x_dim 3")

    y = denorm(x).detach().cpu()
    y = (y * 255.0).round().to(torch.uint8)  # (3, H, W)
    y = y.permute(1, 2, 0).numpy()  # (H, W, 3)
    return Image.fromarray(y, mode="RGB")
