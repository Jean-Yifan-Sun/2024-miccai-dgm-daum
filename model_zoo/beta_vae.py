from typing import Sequence

import torch
import torch.nn.functional as F

from model_zoo.beta_vae_higgings import VAEHigLoss
from net_utils.convolution import TransposedConvolution, Convolution
from net_utils.initialize import *
from net_utils.variational import reparameterize


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class ResidualBlock(nn.Module):
    """
    Residual block with  dropout
    """

    def __init__(self, in_channels: int, out_channels: int, activation, transpose_conv: bool = False, num_layers: int = 3,
                 residual: bool = True, group_norm: int = 0, dimension: int = 2,
                 dropout: float = 0.0, *args, **kwargs):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param num_layers: number of layers in the residual block
        :param kernel_size: kernel size of the convolution
        :param stride: stride of the convolution
        :param residual: whether to use residual connections
        :param group_norm: number of groups for group normalization
        :param dropout: dropout rate
        """
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.residual = residual
        self.group_norm = group_norm
        self.dimension = dimension
        self.dropout = dropout
        self.transpose_conv = transpose_conv


        # initialize layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # first layer
            if i == 0:
                if transpose_conv:
                    self.layers.append(TransposedConvolution(in_channels, out_channels, kernel_size=4, stride=2, padding=1, dimension=dimension))
                else:
                    self.layers.append(Convolution(in_channels, out_channels, kernel_size=4, stride=2, padding=1, dimension=dimension))
            # other layers
            else:
                self.layers.append(Convolution(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dimension=dimension))

        # group normalization
        if group_norm > 0:
            self.group_norms = nn.ModuleList()
            for i in range(num_layers):
                self.group_norms.append(nn.GroupNorm(min(group_norm, out_channels), out_channels))

        # dropout
        if dropout > 0.0:
            self.dropout_layers = nn.ModuleList()
            for i in range(num_layers):
                self.dropout_layers.append(nn.Dropout(dropout))

        # activation
        self.activation = activation

    def forward(self, x):
        """
        Forward pass through the residual block.
        :param x: Input tensor.
        :return: Output tensor.
        """
        # save input for residual connection
        residual = None

        # pass through layers
        for i in range(self.num_layers):
            # convolution
            x = self.layers[i](x)
            if i == 0:
                residual = x

            # group normalization
            if self.group_norm > 0:
                x = self.group_norms[i](x)

            # dropout
            if self.dropout > 0.0:
                x = self.dropout_layers[i](x)

            # activation
            x = self.activation(x)

        if self.residual:
            x = x + residual

        return x


class BetaVAE(nn.Module):
    def __init__(self, image_size: int, image_depth: int, dimension: int, hidden_channels: Sequence[int],
                 z_dim: int = 32, z_channels: int = None, sigmoid: bool = False,
                 activation: str = 'selu', layers_per_block: int = 1, group_norm: int = 0, dropout: float = 0.0):
        """
        :param dim: input dimension (width/height of image)
        :param image_depth: number of input channels
        :param hidden_channels: number of channels in hidden layers
        :param z_dim: latent dimension
        :param activation: activation function
        """
        super(BetaVAE, self).__init__()
        self.image_size = image_size
        self.image_depth = image_depth
        self.dimension = dimension
        self.hidden_channels = hidden_channels
        self.z_dim = z_dim
        self.layers_per_block = layers_per_block
        self.group_norm = group_norm
        self.dropout = dropout
        if z_channels is None:
            self.z_channels = image_depth // 2 if image_depth > 1 else image_depth
        else:
            self.z_channels = z_channels

        self.activation = None
        if activation == 'selu':
            self.activation = nn.SELU(True)
        elif activation == 'relu':
            self.activation = nn.ReLU(True)
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(True)
        else:
            raise NotImplementedError(f'Activation {activation} not implemented')

        ### Encoder ###
        self.encoder = nn.Sequential()
        # Hidden layers reducing spatial dimension by factor 2
        for i, (in_ch, out_ch) in enumerate(zip([self.image_depth if self.dimension == 2 else 1] + list(self.hidden_channels[:-1]), self.hidden_channels)):
            residual_block = ResidualBlock(in_ch, out_ch, activation=self.activation, num_layers=layers_per_block,
                                           residual=True, group_norm=self.group_norm, dropout=self.dropout, dimension=self.dimension)
            self.encoder.add_module(f'residual_block_{i}', residual_block)

        if z_dim > 0:
            # Reduce to 1D representation
            self.encoder_output_size = self.hidden_channels[-1] * (self.image_size // 2 ** len(self.hidden_channels)) ** 2
            self.encoder.add_module('flatten', View((-1, self.encoder_output_size)))
            self.encoder.add_module('fc', nn.Linear(self.encoder_output_size, self.z_dim * 4))
            self.encoder.add_module('act_fc', self.activation)
            self.encoder.add_module('fc_encoder_out', nn.Linear(self.z_dim * 4, self.z_dim * 2))
        else:
            # Remain in image space
            conv_out = Convolution(self.hidden_channels[-1], self.z_channels * 4, kernel_size=3, stride=1, padding=1, dimension=self.dimension)
            conv_out_2 = Convolution(self.z_channels * 4, self.z_channels * 2, kernel_size=3, stride=1, padding=1,
                                     dimension=self.dimension)
            self.encoder.add_module('conv_out', conv_out)
            self.encoder.add_module('act_conv_out', self.activation)
            self.encoder.add_module('conv_out_2', conv_out_2)

        ### Decoder ###
        self.decoder = nn.Sequential()

        if z_dim > 0:
            self.decoder.add_module('fc', nn.Linear(self.z_dim, self.z_dim * 4))
            self.decoder.add_module('act_fc', self.activation)
            self.decoder.add_module('fc_decoder_in', nn.Linear(self.z_dim * 4, self.encoder_output_size))
            self.decoder.add_module('act_decoder_in', self.activation)
            self.decoder.add_module('unflatten', View((-1, self.hidden_channels[-1], self.dim // 2 ** len(self.hidden_channels), self.dim // 2 ** len(self.hidden_channels))))
        else:
            conv_in = Convolution(self.z_channels, self.hidden_channels[-1], kernel_size=3, stride=1, padding=1, dimension=self.dimension)
            conv_in_2 = Convolution(self.hidden_channels[-1], self.hidden_channels[-1], kernel_size=3, stride=1,
                                    padding=1, dimension=self.dimension)
            self.decoder.add_module('conv_in', conv_in)
            self.decoder.add_module('act_conv_in', self.activation)
            self.decoder.add_module('conv_in_2', conv_in_2)

        # Hidden layers increasing spatial dimension by factor 2
        for i, (in_ch, out_ch) in enumerate(
                zip(self.hidden_channels[::-1], list(self.hidden_channels[::-1][1:]) +  [self.hidden_channels[0]])):
            residual_block = ResidualBlock(in_ch, out_ch, transpose_conv=True, activation=self.activation, dimension=self.dimension,
                                           num_layers=layers_per_block, residual=True, group_norm=0, dropout=self.dropout)
            self.decoder.add_module(f'residual_block_{i}', residual_block)

        # Output layer
        conv_out = Convolution(self.hidden_channels[0], self.image_depth if self.dimension == 2 else 1, kernel_size=3, stride=1, padding=1, dimension=self.dimension)
        self.decoder.add_module(f'conv_out', conv_out)

        if sigmoid:
            self.decoder.add_module(f'out_sigmoid', torch.nn.Sigmoid())

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
        if self.dimension == 2 and len(x.shape) != 4:
            raise ValueError(f'Input tensor has shape {x.shape} but expected shape (N,C,H,W)')
        elif self.dimension == 3 and len(x.shape) != 5:
            raise ValueError(f'Input tensor has shape {x.shape} but expected shape (N,C,D,H,W)')

    def reshape_input(self, x):
        if self.dimension == 2 and len(x.shape) == 3:
            x = x.unsqueeze(1)
        elif self.dimension == 3 and len(x.shape) == 4:
            x = x.unsqueeze(1)
        return x

    def reshape_output(self, x, input_shape):
        if self.dimension == 2 and len(input_shape) == 3:
            x = x.squeeze(1)
        elif self.dimension == 3 and len(input_shape) == 4:
            x = x.squeeze(1)
        return x

    def forward(self, x):
        input_shape = x.shape
        x = self.reshape_input(x)

        self._assert_input_shape(x)

        z, dist = self.encode_and_sample(x)
        x_recon = self.decode(z)

        x_recon = self.reshape_output(x_recon, input_shape)
        z = self.reshape_output(z, input_shape)

        return x_recon, z, dist

    def encode(self, x):
        self._assert_input_shape(x)
        return self.encoder(x)

    def decode(self, z):
        self._assert_input_shape(z)
        return self.decoder(z)

    def encode_and_sample(self, x):
        input_shape = x.shape
        x = self.reshape_input(x)

        distributions = self.encode(x)
        if self.z_dim > 0:
            mu = distributions[:, :self.z_dim]
            logvar = distributions[:, self.z_dim:]
        else:
            mu = distributions[:, :self.z_channels, ...]
            logvar = distributions[:, self.z_channels:, ...]
        z = reparameterize(mu, logvar)
        # z = self.reshape_output(z, input_shape)
        return z, {'z_mu': mu, 'z_logvar': logvar}



class AttributeRegularizedBetaVAELoss:

    def __init__(self, kl_beta=4, kl_gamma=10.0, kl_max_capacity=25, attr_gamma=1, attr_factor=1):
        super(AttributeRegularizedBetaVAELoss, self).__init__()
        self.beta_vae_loss = VAEHigLoss(beta=kl_beta, gamma=kl_gamma, max_capacity=kl_max_capacity, loss_type='B')

        self.attr_gamma = attr_gamma
        self.attr_factor = attr_factor

    def __call__(self, x_recon, x, z, dist, labels, reg_dims):
        loss = self.beta_vae_loss(x_recon, x, dist)
        attr_loss = self.attribute_regularization_loss(z, labels, reg_dims, self.attr_gamma, self.attr_factor)
        loss += attr_loss

        return loss

    @staticmethod
    def pairwise_distance_matrix(x):
        # x is a tensor of shape (B, L)
        B, L = x.shape

        # Expand and transpose x to get two tensors of shape (B, B, L)
        x1 = x.unsqueeze(1).expand(B, B, L)
        x2 = x.unsqueeze(0).expand(B, B, L)

        # Compute the pairwise distance for each column
        # Resulting shape will be (B, B, L), then transpose to (L, B, B)
        distance_matrix = (x1 - x2).transpose(0, 2)

        return distance_matrix


    @staticmethod
    def attribute_regularization_loss(z, labels, reg_dims, gamma, factor=1.0):
        """
        Computes the attribute regularization loss for the given latent space
        :param z: torch Variable, (B, N)
        :param labels: torch Variable, (B, L)
        :param reg_dims: set (L, )
        :param gamma: float
        :param factor: parameter for scaling the loss
        :return: Regularization loss, torch Variable
        """
        # compute latent distance matrix
        lc_dist_mat = AttributeRegularizedBetaVAELoss.pairwise_distance_matrix(z[:, reg_dims])

        # compute attribute distance matrix
        attribute_dist_mat = AttributeRegularizedBetaVAELoss.pairwise_distance_matrix(labels)

        lc_tanh = torch.tanh(lc_dist_mat * factor)
        attribute_sign = torch.sign(attribute_dist_mat)
        sign_loss = F.l1_loss(lc_tanh, attribute_sign.float(), reduction='mean')
        return gamma * sign_loss * len(reg_dims)

