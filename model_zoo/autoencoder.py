import logging

from einops import rearrange
import torch
import torch.nn as nn
from torch import einsum

from net_utils.convolution import Convolution
from net_utils.diffusion.denoising_unet import DownBlock, UpBlock
from net_utils.diffusion.normalization import Normalization
from net_utils.variational import reparameterize

from enum import Enum, auto


class Action(Enum):
    ENCODE = auto()
    DECODE = auto()
    FORWARD = auto()


class Encoder(nn.Module):
    def __init__(self, **config):
        super().__init__()
        layer_channels = config['layer_channels']
        layer_multiplier = config['layer_multiplier']
        layer_attention = config['layer_attention']

        self.encoder = nn.Sequential()
        conv_in = Convolution(config['input_size'][0], layer_channels, kernel_size=3, padding=1, stride=1, **config)
        self.encoder.append(conv_in)

        for in_mul, out_mul, attn in zip([1] + layer_multiplier[:-1], layer_multiplier, layer_attention):
            ic = in_mul * layer_channels
            oc = out_mul * layer_channels
            self.encoder.append(DownBlock(in_channels=ic, out_channels=oc, **config))

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, sigmoid_out: bool, **config):
        super().__init__()
        layer_channels = config['layer_channels']
        layer_multiplier = config['layer_multiplier']
        layer_attention = config['layer_attention']

        self.decoder = nn.Sequential()
        ics,ocs = [],[]
        for in_mul, out_mul, attn in zip(layer_multiplier[::-1], layer_multiplier[::-1][1:] + [1],
                                         layer_attention[::-1]):
            ic = in_mul * layer_channels
            oc = out_mul * layer_channels
            self.decoder.append(UpBlock(in_channels=ic, out_channels=oc, **config))
            ics.append(ic)
            ocs.append(oc)
        # logging.info(f'ics: {ics}, ocs: {ocs}')

        out_projection = nn.Sequential(Normalization(in_channels=layer_channels, **config),
                                       nn.SiLU(),
                                       Convolution(layer_channels, config['input_size'][0], kernel_size=3, padding=1,
                                                   stride=1, **config))
        self.decoder.append(out_projection)

        if sigmoid_out:
            self.decoder.append(torch.nn.Sigmoid())

    def forward(self, x):
        return self.decoder(x)


class Autoencoder(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.input_size = config['input_size']
        self.input_dim = config['input_dim']

        self.encoder = Encoder(**config)
        self.decoder = Decoder(**config)

    def encode(self, x: torch.Tensor):
        x = self.reshape_input(x)
        return self.encoder(x)

    def decode(self, x: torch.Tensor):
        x = self.decoder(x)
        return self.reshape_output(x)

    def _forward(self, x: torch.Tensor):
        return self.decode(self.encode(x))

    def forward(self, x: torch.Tensor, action: Action = Action.FORWARD):
        if action == Action.ENCODE:
            return self.encode(x)
        elif action == Action.DECODE:
            return self.decode(x)
        elif action == Action.FORWARD:
            return self._forward(x)
        else:
            raise NotImplementedError

    def reshape_input(self, x):
        if self.input_dim == 2 and len(x.shape) == 3:
            x = x.unsqueeze(1)
        elif self.input_dim == 3 and len(x.shape) == 4:
            x = x.unsqueeze(1)
        return x

    def reshape_output(self, x):
        if self.input_dim == 2:
            x = x.squeeze(1)
        elif self.input_dim == 3:
            x = x.squeeze(1)
        return x


class KLAutoencoder(Autoencoder):
    def __init__(self, **config):
        super(KLAutoencoder, self).__init__(**config)
        self.input_channels = config['input_size'][0]
        layer_channels = config['layer_channels']
        layer_multiplier = config['layer_multiplier']

        self.kl_encoder = nn.Sequential()
        conv_out = Convolution(layer_channels * layer_multiplier[-1], self.input_channels * 4, kernel_size=3,
                               stride=1, padding=1, **config)
        conv_out_2 = Convolution(self.input_channels * 4, self.input_channels * 2, kernel_size=3, stride=1,
                                 padding=1, **config)
        self.kl_encoder.append(conv_out)
        self.kl_encoder.append(nn.SiLU())
        self.kl_encoder.append(conv_out_2)

        self.kl_decoder = nn.Sequential()
        conv_in = Convolution(self.input_channels, self.input_channels * 4, kernel_size=3, stride=1,
                              padding=1, **config)
        conv_in_2 = Convolution(self.input_channels * 4, layer_channels * layer_multiplier[-1], kernel_size=3, stride=1,
                                padding=1, **config)
        self.kl_decoder.append(conv_in)
        self.kl_decoder.append(nn.SiLU())
        self.kl_decoder.append(conv_in_2)

    def encode(self, x: torch.Tensor, dist: bool = False):
        x = super().encode(x)
        x = self.kl_encoder(x)

        mu = x[:, :self.input_channels, ...]
        logvar = x[:, self.input_channels:, ...]

        z = reparameterize(mu, logvar)

        if dist:
            return z, {'z_mu': mu, 'z_logvar': logvar}
        else:
            return z

    def decode(self, x: torch.Tensor):
        x = self.kl_decoder(x)
        return super().decode(x)

    def _forward(self, x: torch.Tensor):
        z, dist = self.encode(x, dist=True)
        return self.decode(z), dist


class VectorQuantizer(nn.Module):
    """
    Vector Quantization Layer taken from:
    https://huggingface.co/spaces/shreeshaaithal/text2art/blob/main/taming-transformers/taming/modules/vqvae/quantize.py
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, vq_size, vq_dim, vq_beta, remap=None, **config):
        super().__init__()
        self.vq_size = vq_size
        self.vq_dim = vq_dim
        self.beta = vq_beta

        self.embedding = nn.Embedding(self.vq_size, self.vq_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.vq_size, 1.0 / self.vq_size)

        self.remap = remap
        self.re_embed = vq_size

    def forward(self, z):
        original_shape = z.shape
        batch_size, num_channels = original_shape[0], original_shape[1]
        spatial_dims = original_shape[2:]  # This will capture H, W or D, H, W

        # Flatten z to the form (batch_size, product of spatial dimensions, num_channels)
        z_permuted = z.permute(0, *range(2, z.dim()), 1).contiguous()  # B, D, H, W, C (or B, H, W, C)
        z_flattened = z_permuted.reshape(-1, num_channels)  # Flatten while preserving the channel structure

        # Compute distances from z to embeddings
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, self.embedding.weight.T)

        # Find the closest embeddings indices
        min_encoding_indices = torch.argmin(d, dim=1)
        # Reshape min_encoding_indices to have the same batch and spatial dimensions as z
        min_encoding_indices = min_encoding_indices.view(batch_size, *spatial_dims)

        # Fetch the corresponding embeddings and reshape
        z_q = self.embedding(min_encoding_indices)
        z_q = z_q.permute(0, -1, *range(1, z_q.dim() - 1))  # Move the channel back to its original place

        # Compute loss
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
               torch.mean((z_q - z.detach()) ** 2)

        # Preserve gradients
        z_q = z + (z_q - z).detach()

        # Reshape back to match original input shape
        z_q = z_q.view(original_shape)

        return z_q, loss

    def get_codebook_entry(self, indices, shape):
        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class VQAutoencoder(Autoencoder):
    def __init__(self, **config):
        super(VQAutoencoder, self).__init__(**config)
        self.input_channels = config['input_size'][0]
        self.input_dim = config['input_dim']
        layer_channels = config['layer_channels']
        layer_multiplier = config['layer_multiplier']
        vq_dim = config['vq_dim']

        self.encoder_out = nn.Sequential(
            Convolution(layer_channels * layer_multiplier[-1], 1, kernel_size=3, stride=1,
                                       padding=1, **config),
            Normalization(1, normalization_groups=1)
            )

        self.quant_conv = Convolution(1, vq_dim, kernel_size=1, stride=1, padding=0, **config)
        self.quantize = VectorQuantizer(vq_beta=0.25, **config)
        self.post_quant_conv = Convolution(vq_dim, layer_channels * layer_multiplier[-1], kernel_size=1, stride=1,
                                           padding=0, **config)

    def encode(self, x: torch.Tensor):
        # logging.info(f'encode input x shape: {x.shape}')
        # input x shape: torch.Size([16, 96, 96, 13])
        x = super().encode(x)
        # logging.info(f'x shape: {x.shape}')
        # x shape: torch.Size([16, 64, 96, 48, 6])
        x = self.encoder_out(x)
        return x

    def decode(self, x: torch.Tensor, loss: bool = False):
        # logging.info(f'decode input x shape: {x.shape}')
        # input x shape: torch.Size([16, 1, 96, 48, 6])
        z = self.quant_conv(x)
        #z = x
        z_shape = z.shape
        # logging.info(f'quant_conv z shape: {z_shape}')
        # z shape: torch.Size([16, 4, 96, 48, 6])
        if self.input_dim == 30:
            assert len(z_shape) == 5
            B, C, D, H, W = z_shape
            z = z.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
            #z = z.squeeze(1)

        z, l = self.quantize(z)
        l = l * 10
        # logging.info(f'quantize z size: {z.shape}')
        if self.input_dim == 30:
            B, C, D, H, W = z_shape
            z = z.reshape(B, D, C, H, W).permute(0, 2, 1, 3, 4)
            # z = z.unsqueeze(1)

        z = self.post_quant_conv(z)
        # logging.info(f'post_quant_conv z size: {z.shape}')
        # z size: torch.Size([16, 64, 96, 48, 6])
        x = super().decode(z)
        # logging.info(f'Decoder z size: {x.shape}')
        # Decoder size: torch.Size([16, 96, 96, 12])
        if loss:
            return x, l
        else:
            return x

    def _forward(self, x: torch.Tensor):
        z = self.encode(x)
        z = self.decode(z, loss=True)
        # logging.info(f'Decoder size: {z.shape}')
        return z



