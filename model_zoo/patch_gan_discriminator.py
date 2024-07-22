import logging

import torch
import torch.nn as nn


# Implementation of the PatchGAN discriminator based on the paper:
# "Image-to-Image Translation with Conditional Adversarial Networks" by Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A.
# Published in 2017 in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
# Link: https://arxiv.org/abs/1611.07004

class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_size=(1, 128, 128), hidden_layers=(8, 8, 8, 8), across_channels=False):
        """
        PatchGAN discriminator class.
        :param input_size: Tuple containing the input size (channels, height, width).
        :param hidden_layers: List containing the number of channels in each hidden layer.
        """
        super(PatchGANDiscriminator, self).__init__()
        self.across_channels = across_channels
        in_channels = 1
        if across_channels:
            in_channels = input_size[0]

        layers = [nn.Conv2d(in_channels, hidden_layers[0], kernel_size=4, stride=2, padding=1),
                  nn.LeakyReLU(0.2, inplace=True)]

        for i in range(1, len(hidden_layers)):
            in_channels = hidden_layers[i - 1]
            out_channels = hidden_layers[i]
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(hidden_layers[-1], 1, kernel_size=1, stride=1, padding=0))

        self.model = nn.Sequential(*layers)

        # Create a dummy input to calculate the output size
        dummy_input = torch.zeros(input_size).unsqueeze(0)
        self.output_size = None
        with torch.no_grad():
            self.output_size = self(dummy_input).squeeze(0).shape

        # Print the output size
        print(f'PatchGAN discriminator output size: {self.output_size}')
        logging.info(f"INFO:root:[PatchGANDiscriminator]: Output size {self.output_size}'")

    def forward(self, img):
        if self.across_channels:
            return self.model(img)
        else:
            B, C, H, W = img.shape
            # Reshape to only have one channel
            img = img.view(-1, 1, H, W)

            # Pass through the model
            output = self.model(img)

            # Reshape to match the original batch size
            output = output.view(B, C, *output.shape[2:])

            # Take the mean across the channels
            return output# .mean(dim=1)

    def generate_ground_truth(self, labels, device):
        """
        Generate ground truth for PatchGAN discriminator training.
        :param labels: tensor of 0 or 1 values indicating whether the image is real or fake.
        :param device: The device to create the tensor on (e.g., 'cuda' or 'cpu').
        :return: A tensor filled with ones or zeros matching the output size of the discriminator.
        """
        # Ensure the labels tensor is in the correct shape (batch_size, 1)
        labels = labels.view(-1, 1)

        # Repeat the labels to match the spatial dimensions of the discriminator's output
        # The new shape will be [batch_size, C, _,  _]
        repeated_labels = labels.repeat(*self.output_size)

        # Expand the dimensions to match the output size of the discriminator
        ground_truth = repeated_labels.view(-1, *self.output_size)

        return ground_truth.to(device)

