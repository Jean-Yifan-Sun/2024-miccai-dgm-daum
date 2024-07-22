import torch
import torch.nn as nn


class Normalization(nn.Module):
    """
    Normalization Layer
    Either Batch Normalization (Wide Residual Networks) or Group Normalization (DDPM)
    """
    def __init__(self, in_channels, normalization: str = 'group', normalization_groups: int = 32, **config):
        super().__init__()
        if normalization == 'batch':
            self.norm = nn.BatchNorm2d(in_channels)
        elif normalization == 'group':
            self.norm = nn.GroupNorm(num_groups=min(in_channels, normalization_groups), num_channels=in_channels)
        elif normalization == 'none':
            self.norm = nn.Identity()
        else:
            raise ValueError(f'Unknown normalization {normalization}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)
