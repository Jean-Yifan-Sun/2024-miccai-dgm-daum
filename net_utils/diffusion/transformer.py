import torch
import torch.nn as nn

from net_utils.convolution import Convolution
from net_utils.diffusion.attention import SpatialSelfAttention, SpatialCrossAttention, DepthwiseSelfAttention
from net_utils.diffusion.normalization import Normalization


class MLP(nn.Module):
    """
    Multi-layer perceptron to be used in the spatial transformer. Applies a series of linear layers with SiLU activation
    and dropout.
    """
    def __init__(self, channels: int, dropout: float = 0.1, **config):
        super().__init__()
        self.norm = Normalization(channels, **config)
        self.layers = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x: Input tensor of shape (B, C, *).
        """
        residual = x
        x = self.norm(x)
        x = x.transpose(1, -1)
        x = self.layers(x)
        x = x.transpose(1, -1)
        return x + residual


class BasicSpatialTransformer(nn.Module):
    """
    Basic spatial transformer similar to High Resolution Image Synthesis with Latent Diffusion Models.

    Architecture:
        LayerNorm   | Not used as we already have ResidualBlocks
        Conv1x1     | Not used as we already have ResidualBlocks
        Reshape     | Not required as done by SpatialAttention Blocks
        N times:
            SelfAttention     | Basic
            MLP               | SpactialTransformer
            CrossAttention    | Block
        Reshape     | Not required as done by SpatialAttention Blocks
        Conv1x1     | Not used as we already have ResidualBlocks
    """

    def  __init__(self, channels: int, transformer_blocks: int, **config):
        super().__init__()
        self.blocks = nn.ModuleList([BasicSpatialTransformerBlock(channels, **config)
                                     for _ in range(transformer_blocks)])

    def forward(self, x: torch.Tensor, context: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        x_in = x
        for block in self.blocks:
            x = block(x, context, mask)
        return x + x_in

class BasicSpatialTransformerBlock(nn.Module):
    """
    Basic spatial transformer block. Uses spatial self-attention and spatial cross-attention.

    Architecture:
        SelfAttention
        MLP
        CrossAttention
    """

    def __init__(self, channels: int, context_dim: int = None, **config):
        super().__init__()

        self.spatial_attention = SpatialSelfAttention(channels, **config)
        self.cross_attention = SpatialCrossAttention(channels, context_dim, **config) if context_dim is not None and context_dim > 0 else None
        self.mlp = MLP(channels, **config)

        if config['input_dim'] == 3:
            self.depthwise_attention = DepthwiseSelfAttention(channels, **config)
        else:
            self.depthwise_attention = None

    def forward(self, x: torch.Tensor, context: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.spatial_attention(x)
        if self.depthwise_attention is not None:
            x = self.depthwise_attention(x)
        x = self.mlp(x)
        if self.cross_attention is not None and context is not None:
            x = self.cross_attention(x, context, mask)
        return x

