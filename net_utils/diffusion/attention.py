import math
import torch
import torch.nn as nn
import logging

from net_utils.diffusion.normalization import Normalization


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Attention(nn.Module):
    """
    General attention implementation
    """

    def __init__(self, query_dim: int, context_dim: int = None, attention_heads: int = 1, attention_head_dim: int = 16,
                 dropout: float = 0.1, **config):
        super().__init__()
        self.attention_heads = attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = attention_heads * attention_head_dim
        self.scale = attention_head_dim ** -0.5

        context_dim = context_dim or query_dim

        self.q = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.k = nn.Linear(context_dim, self.inner_dim, bias=False)
        self.v = nn.Linear(context_dim, self.inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, query_dim),
            nn.Dropout(dropout)
        )


    def forward(self, query: torch.Tensor, context: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if context is None:
            context = query
        *query_batch_dims, query_seq_len, _ = query.size()
        *context_batch_dims, context_seq_len, _ = context.size()

        q = self.q(query).view(*query_batch_dims, query_seq_len, self.attention_heads,
                               self.attention_head_dim).transpose(-3, -2)
        k = self.k(context).view(*context_batch_dims, context_seq_len, self.attention_heads,
                                 self.attention_head_dim).transpose(-3, -2)
        v = self.v(context).view(*context_batch_dims, context_seq_len, self.attention_heads,
                                 self.attention_head_dim).transpose(-3, -2)

        attention_scores = torch.einsum('...sd,...kd->...sk', q, k) * self.scale

        if mask is not None:
            expanded_mask = mask.unsqueeze(-2)  # Add a dimension for attention heads
            while len(expanded_mask.shape) < len(attention_scores.shape):
                expanded_mask = expanded_mask.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(expanded_mask == 0, float('-inf'))

        attention = attention_scores.softmax(dim=-1)

        out = torch.einsum('...sk,...kd->...sd', attention, v)
        out = out.reshape(*query_batch_dims, query_seq_len, self.inner_dim)

        return self.to_out(out)

class SelfAttention(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other. Uses three q, k, v linear layers to
    compute attention. Follows the attention block from Denoising Diffusion Probabilistic Models.

    Args:
        query_dim: Dimension of the query
        heads: Number of attention heads
        head_dim: Dimension of each head
        dropout: Dropout probability
    """
    def __init__(self, query_dim: int,
                 **config):
        super().__init__()
        self.attention = Attention(query_dim, query_dim, **config)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        return self.attention(x, x, mask)


class SpatialSelfAttention(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other. Uses three q, k, v linear layers to
    compute attention. Follows the attention block from High-Resolution Image Synthesis with Latent Diffusion Models.

    Args:
        query_dim: Dimension of the query
        heads: Number of attention heads
        head_dim: Dimension of each head
        dropout: Dropout probability
    """
    def __init__(self, query_dim: int, **config):
        super().__init__()
        self.attention = Attention(query_dim, query_dim, **config)
        self.norm = Normalization(query_dim, **config)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        input_dim = len(x.shape)
        if input_dim not in [4, 5]:
            raise ValueError("Input tensor must be 4D or 5D.")

        residual = x
        x = self.norm(x)

        if input_dim == 4:  # For 4D tensors, straightforward application over H and W dimensions.
            B, C, H, W = x.shape
            x = x.view(B, C, H * W).transpose(1, 2)  # Shape: B, H*W, C
            x = self.attention(x, x)
            x = x.transpose(1, 2).reshape(B, C, H, W)
        elif input_dim == 5:  # For 5D tensors, apply attention over H and W, keeping D separate.
            B, C, D, H, W = x.shape
            x = x.permute(0, 2, 3, 4, 1).reshape(B, D, H * W, C)  # Reshape to (B, D, H*W, C) for attention.
            x = self.attention(x, x)
            x = x.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)  # Reshape back to (B, C, D, H, W).

        x = x + residual
        return x


class DepthwiseSelfAttention(nn.Module):
    """
    An attention block that allows different depths to attend to each other following Denoising
    Diffusion Probabilistic Models for 3D medical image generation.
    https://arxiv.org/abs/2211.03364
    Uses three q, k, v linear layers to compute attention.

    Args:
        query_dim: Dimension of the query
        heads: Number of attention heads
        head_dim: Dimension of each head
        dropout: Dropout probability
    """
    def __init__(self, query_dim: int, depth_embedding: bool, **config):
        super().__init__()

        self.attention = Attention(query_dim, query_dim, **config)
        self.norm = Normalization(query_dim, **config)
        if depth_embedding:
            self.sinusoidal_embedding = SinusoidalPosEmb(query_dim)
        else:
            self.sinusoidal_embedding = None

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError("Input tensor must be 5D for depthwise attention.")
        B, C, D, H, W = x.shape

        residual = x
        x = self.norm(x)

        # Apply sinusoidal embedding if enabled.
        if self.sinusoidal_embedding is not None:
            depth_emb = self.sinusoidal_embedding(torch.arange(D, device=x.device))
            depth_emb = depth_emb.permute(1, 0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # Now [1, C, D, 1, 1]

            x += depth_emb

        # Reshape and permute x for attention: treat each depth as a sequence.
        x = x.permute(0, 3, 4, 2, 1).reshape(B, H * W, D, C)  # B, H*W, D, C

        # Apply attention.
        x = self.attention(x, x)

        # Reshape x back to original dimensions.
        x = x.view(B, H, W, D, C).permute(0, 4, 3, 1, 2)  # B, C, D, H, W

        x += residual
        return x


class CrossAttention(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other. Uses three q, k, v linear layers to
    compute attention. Follows the attention block from Denoising Diffusion Probabilistic Models.

    Args:
        query_dim: Dimension of the query
        context_dim: Dimension of the context
        heads: Number of attention heads
        head_dim: Dimension of each head
        dropout: Dropout probability
    """
    def __init__(self, query_dim: int, context_dim: int, **config):
        super().__init__()
        self.attention = Attention(query_dim, context_dim, **config)

    def forward(self, query: torch.Tensor, context: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        return self.attention(query, context, mask)


class SpatialCrossAttention(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other. Uses three q, k, v linear layers to
    compute attention. Follows the attention block from High-Resolution Image Synthesis with Latent Diffusion Models.
    """
    def __init__(self, query_dim: int, context_dim: int, **config):
        super().__init__()
        self.attention = Attention(query_dim, context_dim, **config)
        self.norm = Normalization(query_dim, **config)

    def forward(self, x: torch.Tensor, context: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x_dim = x.dim()
        if x_dim not in [4, 5]:
            raise ValueError("Input tensor must be 4D or 5D.")
        B, C, *spatial_dims = x.shape  # Supports both 4D and 5D tensors.
        residual = x

        x = self.norm(x)

        # Handle the 4D and 5D cases with spatial dimensions reshaped for attention.
        if x_dim == 4:
            H, W = spatial_dims
            x = x.view(B, C, H * W).transpose(1, 2)  # B, H*W, C
        elif x_dim == 5:
            D, H, W = spatial_dims
            x = x.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, C)  # B, D*H*W, C

        # Apply attention.
        x = self.attention(x, context, mask)

        # Reshape x back to original dimensions.
        if x_dim == 4:
            x = x.transpose(1, 2).reshape(B, C, H, W)
        elif x_dim == 5:
            x = x.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)

        x += residual
        return x
