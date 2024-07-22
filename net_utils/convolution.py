import torch.nn as nn

class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True, input_dim=2, **config):
        super(Convolution, self).__init__()
        self.conv = None
        if input_dim == 2:
            self.conv = nn.Conv2d
        elif input_dim == 3:
            self.conv = nn.Conv3d
            if type(kernel_size) == int:
                kernel_size = (1, kernel_size, kernel_size)
                stride = (1, stride, stride)
                padding = (0, padding, padding)
        else:
            raise ValueError('Invalid input_dim {}. Must be 2 or 3.'.format(input_dim))

        self.conv = self.conv(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        return self.conv(x)


class TransposedConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True, input_dim=2, **config):
        super(TransposedConvolution, self).__init__()
        self.conv = None
        if input_dim == 2:
            self.conv = nn.ConvTranspose2d
        elif input_dim == 3:
            self.conv = nn.ConvTranspose3d
            if type(kernel_size) == int:
                kernel_size = (1, kernel_size, kernel_size)
                stride = (1, stride, stride)
                padding = (0, padding, padding)
        else:
            raise ValueError('Invalid input_dim {}. Must be 2 or 3.'.format(input_dim))

        self.conv = self.conv(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        return self.conv(x)

class Dropout(nn.Module):
    def __init__(self, dropout=0.1, **config):
        super(Dropout, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x)
