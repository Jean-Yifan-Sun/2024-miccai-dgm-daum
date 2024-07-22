from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_zoo.autoencoder import Encoder
from model_zoo.beta_vae_higgings import VAEHigLoss
from model_zoo.beta_vae import ResidualBlock, View
from net_utils.initialize import *
from net_utils.variational import reparameterize

class CardiacClassifier(nn.Module):
    def __init__(self, **config):
        """
        :param dim: input dimension (width/height of image)
        :param image_depth: number of input channels
        :param hidden_channels: number of channels in hidden layers
        :param z_dim: latent dimension
        :param activation: activation function
        """
        super(CardiacClassifier, self).__init__()
        layer_channels = config['layer_channels']
        layer_multiplier = config['layer_multiplier']
        input_size = config['input_size']
        self.input_dim = config['input_dim']

        self.encoder = Encoder(**config)
        self.classifier_head = nn.Sequential()

        self.encoder_output_size = layer_channels * layer_multiplier[-1] * ((input_size[-1] // 2 ** len(layer_multiplier)) ** 2) * input_size[-3]
        self.classifier_head.append(View((-1, self.encoder_output_size)))
        self.classifier_head.append(nn.Linear(self.encoder_output_size, 32))
        self.classifier_head.append(nn.SiLU())
        self.classifier_head.append(nn.Linear(32, 1))

        self.weight_init()

    def weight_init(self):
        for block in self._modules.values():
            if hasattr(block, '__iter__'):  # Check if the block is iterable
                for m in block:
                    kaiming_init(m)
            else:
                kaiming_init(block)

    def _assert_input_shape(self, x):
        # target shape for dimension=2 (N,C,H,W)
        # target shape for dimension=3 (N,C,D,H,W)
        if self.input_dim == 2 and len(x.shape) != 4:
            raise ValueError(f'Input tensor has shape {x.shape} but expected shape (N,C,H,W)')
        elif self.input_dim == 3 and len(x.shape) != 5:
            raise ValueError(f'Input tensor has shape {x.shape} but expected shape (N,C,D,H,W)')

    def reshape_input(self, x):
        if self.input_dim == 2 and len(x.shape) == 3:
            x = x.unsqueeze(1)
        elif self.input_dim == 3 and len(x.shape) == 4:
            x = x.unsqueeze(1)
        return x

    def reshape_output(self, x, input_shape):
        if self.input_dim == 2 and len(input_shape) == 3:
            x = x.squeeze(1)
        elif self.input_dim == 3 and len(input_shape) == 4:
            x = x.squeeze(1)
        return x

    def forward(self, x):
        x = self.reshape_input(x)
        self._assert_input_shape(x)

        x = self.encoder(x)
        x = self.classifier_head(x)

        return x.squeeze(1)
