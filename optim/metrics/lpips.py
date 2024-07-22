import torch
import torch.nn as nn
import lpips
from torchvision import transforms


class LPIPS(nn.Module):
    """
    LPIPS metric using AlexNet backbone
    """

    def __init__(self, normalize: bool = True):
        """
        :param normalize: whether to normalize from [0, 1] to [-1, 1]
        """
        super().__init__()
        self.name = 'FID'
        self.device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

        self.lpips_alex = lpips.LPIPS(net='alex').to(self.device)
        self.normalize = normalize

    def forward(self, images_real: torch.Tensor, images_fake: torch.Tensor):
        """
        Calculate the FID score
        images_real: real images with shape (batch_size, channels, height, width)
        images_fake: fake images with shape (batch_size, channels, height, width)
        """
        images_real = images_real.to(self.device)
        images_fake = images_fake.to(self.device)

        if images_real.shape != images_fake.shape:
            raise ValueError("Real and fake images must have same dimension")

        if len(images_real.shape) != 4:
            raise ValueError("Input tensors must have 4 dimensions: B x C x H x W")

        # if images have unequal 1 or 3 channels then we are dealing with a 3D image
        if images_real.shape[1] not in (1, 3):
            # reshape depth dim as extension of batch dim.
            # Treat slices as independent grey scale images.
            B, C, H, W = images_real.shape
            images_real = images_real.reshape(B * C, 1, H, W)
            images_fake = images_fake.reshape(B * C, 1, H, W)

        # if images have only one channel (grayscale) repeat it 3 times
        if images_real.shape[1] == 1:
            images_real = images_real.repeat(1, 3, 1, 1)
            images_fake = images_fake.repeat(1, 3, 1, 1)

        return self.lpips_alex(images_real, images_fake, normalize=self.normalize)
