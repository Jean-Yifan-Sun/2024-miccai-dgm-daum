import torch
import torch.nn as nn

from net_utils.convolution import Convolution, Dropout
from net_utils.diffusion.normalization import Normalization


class ResidualBlock(nn.Module):
    """
    ResNet Block following Wide Residual Networks.
    https://arxiv.org/pdf/1605.07146.pdf
    """
    def __init__(self, in_channels, out_channels, **config):
        super().__init__()
        kernel_size = 3     # Original paper uses 2 consecutive 3x3 kernels. More convolutions was found to be suboptimal.
        padding = kernel_size // 2
        stride = 1

        self.conv1 = Convolution(in_channels, out_channels, kernel_size, stride, padding, **config)
        self.conv2 = Convolution(out_channels, out_channels, kernel_size, stride, padding, **config)
        self.norm1 = Normalization(in_channels, **config)
        self.norm2 = Normalization(out_channels, **config)
        self.dropout = Dropout(**config)
        self.activation = nn.SiLU()               # Original paper uses ReLU

        temb_channels = config['temb_channels'] if 'temb_channels' in config else None
        self.temb_proj = nn.Linear(temb_channels, out_channels) if temb_channels is not None else None

        # Projection shortcut
        if in_channels != out_channels:
            self.projection = Convolution(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, **config)
        else:
            self.projection = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor = None) -> torch.Tensor:
        # Norm-Act-Conv order follows Wide Residual Networks
        identity = x
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv1(x)

        x = self.dropout(x)
        if emb is not None:
            emb = self.temb_proj(emb)
            expander = list(emb.shape) + ([1] * (len(x.shape) - len(emb.shape)))
            x += emb.view(*expander)

        x = self.norm2(x)
        x = self.activation(x)
        x = self.conv2(x)

        proj = self.projection(identity)

        x = x + proj
        return x