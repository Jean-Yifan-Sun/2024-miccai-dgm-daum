from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from model_zoo.autoencoder import Encoder,Decoder
from model_zoo.beta_vae_higgings import VAEHigLoss
from model_zoo.beta_vae import ResidualBlock, View
from net_utils.initialize import *
from net_utils.variational import reparameterize

class CardiacSegmentation3D(nn.Module):
    def __init__(self, **config):
        """
        :param dim: input dimension (width/height of image)
        :param image_depth: number of input channels
        :param hidden_channels: number of channels in hidden layers
        :param z_dim: latent dimension
        :param activation: activation function
        """
        super(CardiacSegmentation3D, self).__init__()
        layer_channels = config['layer_channels']
        layer_multiplier = config['layer_multiplier']
        input_size = config['input_size']
        self.input_dim = config['input_dim']

        self.encoder = Encoder(**config)
        self.decoder = Decoder(**config)
        # self.classifier_head = nn.Sequential()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(input_size[0], 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.bpool = nn.MaxPool3d(kernel_size=2, stride=2, padding=1)

        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1024, 512)

        self.upconv3 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.out_conv = nn.Conv3d(64, input_size[0], kernel_size=1)

        
        self.encoder_output_size = layer_channels * layer_multiplier[-1] * ((input_size[-1] // 2 ** len(layer_multiplier)) ** 2) * input_size[-3]

        # self.classifier_head.append(View((-1, self.encoder_output_size)))
        # self.classifier_head.append(nn.Linear(self.encoder_output_size, 32))
        # self.classifier_head.append(nn.SiLU())
        # self.classifier_head.append(nn.Linear(32, 1))

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
    
    
    def pad_to(self, x, stride):
        c, h, w = x.shape[-3:]

        if c % stride > 0:
            new_c = c + stride - c % stride
        else:
            new_c = c
        if h % stride > 0:
            new_h = h + stride - h % stride
        else:
            new_h = h
        if w % stride > 0:
            new_w = w + stride - w % stride
        else:
            new_w = w
        lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
        lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
        lc, uc = int((new_c-c) / 2), int(new_c-c) - int((new_c-c) / 2)
        pads = (lw, uw, lh, uh, lc, uc)

        # zero-padding by default.
        # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
        out = F.pad(x, pads, "constant", 0)
        self.pads = pads
        return out, pads

    def unpad(self, x, pad):
        if pad[4] + pad[5] > 0:
            x = x[:, :, pad[4]:-pad[5], :, :]
        if pad[2] + pad[3] > 0:
            x = x[:, :, :, pad[2]:-pad[3], :]
        if pad[0] + pad[1] > 0:
            x = x[:, :, :, :, pad[0]:-pad[1]]
        return x


    def forward(self, x):
        # x = self.reshape_input(x)
        # self._assert_input_shape(x)

        # x = self.encoder(x)
        # x = self.decoder(x)
        
        x = self.reshape_input(x)
        self._assert_input_shape(x)

        x, pads = self.pad_to(x, 16)
        # logging.info(f'x shape: {x.shape}') 
        # torch.Size([32, 1, 13, 96, 96])
        e1 = self.enc1(x)
        # logging.info(f'e1 shape: {e1.shape}') 
        # torch.Size([32, 64, 13, 96, 96])
        e2 = self.enc2(self.pool(e1))
        # logging.info(f'e2 shape: {e2.shape}') 
        # torch.Size([32, 128, 6, 48, 48])
        # torch.Size([32, 128, 7, 49, 49])
        e3 = self.enc3(self.pool(e2)) 
        # logging.info(f'e3 shape: {e3.shape}') 
        # torch.Size([32, 256, 3, 24, 24])
        # torch.Size([32, 256, 4, 25, 25])
        e4 = self.enc4(self.pool(e3))
        # logging.info(f'e4 shape: {e4.shape}')
        # torch.Size([32, 512, 3, 13, 13])

        b = self.bottleneck(self.pool(e4))
        # logging.info(f'b shape: {b.shape}')
        # torch.Size([32, 1024, 2, 7, 7])

        d4 = self.upconv4(b)
        # logging.info(f'd4 shape: {d4.shape}')
        # torch.Size([32, 512, 4, 14, 14])

        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)
        out = self.unpad(out, pads)
        # x = self.classifier_head(x)

        return out.squeeze(1)


class CardiacSegmentation2D(nn.Module):
    def __init__(self, **config):
        """
        :param dim: input dimension (width/height of image)
        :param image_depth: number of input channels
        :param hidden_channels: number of channels in hidden layers
        :param z_dim: latent dimension
        :param activation: activation function
        """
        super(CardiacSegmentation2D, self).__init__()
        layer_channels = config['layer_channels']
        layer_multiplier = config['layer_multiplier']
        self.input_size = config['input_size']
        self.input_dim = config['input_dim']

        self.encoder = Encoder(**config)
        self.decoder = Decoder(**config)

        class DoubleConv(nn.Module):
            def __init__(self, in_channels, out_channels):
                super(DoubleConv, self).__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )

            def forward(self, x):
                return self.conv(x)
            
        class DownBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super(DownBlock, self).__init__()
                self.down = nn.Sequential(
                    DoubleConv(in_channels, out_channels),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )

            def forward(self, x):
                return self.down(x)
            
        class UpBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super(UpBlock, self).__init__()
                self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
                self.conv = DoubleConv(in_channels, out_channels)

            def forward(self, x, skip_connection):
                x = self.up(x)
                # Crop and concatenate the skip connection to match dimensions
                diffY = skip_connection.size(2) - x.size(2)
                diffX = skip_connection.size(3) - x.size(3)
                x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
                x = torch.cat([skip_connection, x], dim=1)
                return self.conv(x)

        class FinalConv(nn.Module):
            def __init__(self, in_channels, out_channels):
                super(FinalConv, self).__init__()
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

            def forward(self, x):
                return self.conv(x)

        self.enc1 = DoubleConv(self.input_size[0], 64)
        self.enc2 = DownBlock(64, 128)
        self.enc3 = DownBlock(128, 256)
        self.enc4 = DownBlock(256, 512)

        self.bottom = DoubleConv(512, 1024)

        self.dec4 = UpBlock(1024, 512)
        self.dec3 = UpBlock(512, 256)
        self.dec2 = UpBlock(256, 128)
        self.dec1 = UpBlock(128, 64)

        self.final = FinalConv(64, self.input_size[0])
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
        if len(x.shape)==4:
            x = x.squeeze(0)
            x = x.unsqueeze(1)
        return x

    def reshape_output(self, x, input_shape):
        if self.input_dim == 2 and len(input_shape) == 3:
            x = x.squeeze(1)
        elif self.input_dim == 3 and len(input_shape) == 4:
            x = x.squeeze(1)
        return x        

    def forward(self, x):
        # x = self.reshape_input(x)
        # self._assert_input_shape(x)
        # # Encoder
        # slices = []
        # for i in range(self.input_size[1]):
        #     slice.append(x[:,:,i,:,:])
        # slices = torch.cat(slices,dim=0)
        assert len(x.shape) == 3
        x = x.unsqueeze(1)
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Bottleneck
        bottleneck = self.bottom(enc4)

        # Decoder
        dec4 = self.dec4(bottleneck, enc4)
        dec3 = self.dec3(dec4, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2, enc1)

        # Final output
        out = self.final(dec1)
        return out.squeeze()
