from typing import List

import math
import torch
import torch.nn as nn

from net_utils.convolution import Convolution, TransposedConvolution
from net_utils.diffusion.normalization import Normalization
from net_utils.diffusion.residual_block import ResidualBlock
from net_utils.diffusion.transformer import BasicSpatialTransformer
from net_utils.diffusion.attention import SinusoidalPosEmb


class UpSample(nn.Module):
    """
    Upsampling Layer following PixelCNN++ using a strided transpose convolution.
    Upsamples by a factor of 2.
    https://arxiv.org/pdf/1701.05517.pdf
    """

    def __init__(self, in_channels: int, out_channels: int, sample_resnet: bool = False, **config):
        super().__init__()
        self.upsample = TransposedConvolution(in_channels, out_channels, kernel_size=2, stride=2, **config)
        if sample_resnet:
            self.resnet = ResidualBlock(out_channels, out_channels, **config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if hasattr(self, 'resnet'):
            x = self.resnet(x)
        return x


class DownSample(nn.Module):
    """
    Downsampling Layer following PixelCNN++ using a strided convolution.
    Downsamples by a factor of 2.
    https://arxiv.org/pdf/1701.05517.pdf
    """

    def __init__(self, in_channels: int, out_channels: int, sample_resnet: bool = False, **config):
        super().__init__()
        self.downsample = Convolution(in_channels, out_channels, kernel_size=2, stride=2, **config)
        if sample_resnet:
            self.resnet = ResidualBlock(out_channels, out_channels, **config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        if hasattr(self, 'resnet'):
            x = self.resnet(x)
        return x


class UpBlock(nn.Module):
    """
    UpBlock consiting of ResNetLayers, SpatialTransformer and Upsampling.
    """

    def __init__(self, in_channels: int, out_channels: int, attention: bool = False, **config):
        super().__init__()

        self.resnet_layers = nn.ModuleList([ResidualBlock(in_channels if i == 0 else out_channels, out_channels,
                                                          **config) for i in range(config['residual_blocks'])])

        if attention:
            # One transformer after each residual blocks
            self.transformer_layers = nn.ModuleList([
                BasicSpatialTransformer(out_channels, **config) for _ in range(config['residual_blocks'])])
        else:
            self.transformer_layers = [None for _ in range(config['residual_blocks'])]

        self.upsample = UpSample(out_channels, out_channels, **config)

    def forward(self, x: torch.Tensor, emb: torch.Tensor = None, context: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        for resnet, transformer in zip(self.resnet_layers, self.transformer_layers):
            x = resnet(x, emb)
            if transformer is not None:
                x = transformer(x, context, mask)

        x = self.upsample(x)
        return x


class DownBlock(nn.Module):
    """
    DownBlock consiting of ResNetLayers, SpatialTransformer and Downsampling.
    """

    def __init__(self, in_channels: int, out_channels: int, attention: bool = False, **config):
        super().__init__()
        self.residual_blocks = nn.ModuleList([ResidualBlock(in_channels if i == 0 else out_channels, out_channels,
                                                            **config) for i in range(config['residual_blocks'])])

        if attention:
            # One transformer after each residual blocks
            self.transformer_layers = nn.ModuleList([
                BasicSpatialTransformer(out_channels, **config) for _ in range(config['residual_blocks'])])
        else:
            self.transformer_layers = [None for _ in range(config['residual_blocks'])]

        self.downsample = DownSample(out_channels, out_channels, **config)

    def forward(self, x: torch.Tensor, emb: torch.Tensor = None, context: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        for resnet, transformer in zip(self.residual_blocks, self.transformer_layers):
            x = resnet(x, emb)
            if transformer is not None:
                x = transformer(x, context, mask)

        x = self.downsample(x)
        return x


class MiddleBlock(nn.Module):
    """
    MiddleBlock consisting of:
     ResNetLayer
     SpatialTransformer
     ResNetLayer

    Dimensions are preserved.
    """

    def __init__(self, in_channels: int, **config):
        super().__init__()
        self.resnet_layer1 = ResidualBlock(in_channels, in_channels, **config)
        self.transformer_layer = BasicSpatialTransformer(in_channels, **config)
        self.resnet_layer2 = ResidualBlock(in_channels, in_channels, **config)

    def forward(self, x: torch.Tensor, emb: torch.Tensor = None, context: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.resnet_layer1(x, emb)
        x = self.transformer_layer(x, context, mask)
        x = self.resnet_layer2(x, emb)
        return x


class DenoisingUnet(nn.Module):
    """
        Denoising U-Net similar to High-Resolution Image Synthesis with Latent Diffusion Models.
    """

    def __init__(self, layer_channels: int, layer_multiplier: List[int], layer_attention: List[bool], **config):
        super().__init__()
        if len(layer_multiplier) != len(layer_attention):
            raise ValueError(
                f'Number of resolutions ({len(layer_multiplier)}) must be equal to attention indicators ({len(layer_attention)})')

        self.temb_embedding = SinusoidalPosEmb(config['temb_channels'])

        self.conv_in = Convolution(config['input_size'][0], layer_channels,
                                   kernel_size=3, padding=1, stride=1, **config)

        # Down
        self.down_blocks = nn.ModuleList()
        for in_mul, out_mul, attn in zip([1] + layer_multiplier[:-1], layer_multiplier, layer_attention):
            ic = in_mul * layer_channels
            oc = out_mul * layer_channels

            self.down_blocks.append(DownBlock(ic, oc, attn, **config))

        # Middle
        self.middle_block = MiddleBlock(layer_multiplier[-1] * layer_channels, **config)

        # Up
        self.up_blocks = nn.ModuleList()
        for in_mul, out_mul, attn in zip(layer_multiplier[::-1], layer_multiplier[::-1][1:] + [1], layer_attention[::-1]):
            ic = in_mul * layer_channels
            oc = out_mul * layer_channels

            self.up_blocks.append(UpBlock(ic, oc, attn, **config))

        self.out_projection = nn.Sequential(Normalization(layer_channels, **config),
                                            nn.SiLU(),
                                            Convolution(layer_channels, config['input_size'][0], kernel_size=3, padding=1,
                                                        stride=1, **config)
                                            )

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.conv_in(x)
        x_res = x
        emb = self.temb_embedding(t)
        skip_connections = []

        for down_block in self.down_blocks:
            x = down_block(x, emb, context, mask)
            skip_connections.append(x)

        skip_connections = skip_connections[::-1]
        x = self.middle_block(x, emb, context, mask)

        for i, (up_block, skip) in enumerate(zip(self.up_blocks, skip_connections)):
            x = x + skip
            x = up_block(x, emb, context, mask)

        x = self.out_projection(x + x_res)
        return x
