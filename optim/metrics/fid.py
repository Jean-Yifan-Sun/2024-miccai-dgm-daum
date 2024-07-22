import torch
import torch.nn as nn
from torchvision import transforms


class FrechetInceptionDistance(nn.Module):
    """
    Frechet Inception Distance (FID) metric for evaluating Generative Models
    """

    def __init__(self):
        super().__init__()
        self.name = 'FID'
        self.device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
        self.inception_model = self._load_inception_model()
        self.inception_model.to(self.device)
        self.inception_model.eval()
        self.mu_real = None
        self.sigma_real = None

        self.preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.activations_real = []
        self.activations_fake = []

    def _load_inception_model(self):
        """
        Load the inception model
        """
        from torchvision.models.inception import inception_v3
        inception_model = inception_v3(pretrained=True, transform_input=False)
        inception_model.fc = nn.Identity()
        return inception_model

    def _get_activations(self, images):
        """
        Get activations from the inception model
        """
        with torch.no_grad():
            activations = self.inception_model(images)
            return activations

    def _get_statistics(self, activations):
        """
        Get statistics from the activations
        """
        mu = activations.mean(dim=0)
        sigma = torch.cov(activations.t())
        return mu, sigma

    def _get_fid(self, mu_real, sigma_real, mu_fake, sigma_fake):
        """
        Get the FID score
        """
        from torch.nn.functional import l1_loss
        l_1 = l1_loss(mu_real, mu_fake)

        # Using SVD for numerical stability
        U, S, V = torch.svd(sigma_real @ sigma_fake)
        sqrt = U @ torch.diag(torch.sqrt(S)) @ V.t()

        fid = l_1 + torch.trace(sigma_real + sigma_fake - 2 * sqrt)

        return fid

    def _get_mu_sigma(self, images):
        """
        Get mu and sigma from the images
        """
        activations = self._get_activations(images)
        mu, sigma = self._get_statistics(activations)
        return mu, sigma

    def forward(self, images_real: torch.Tensor, images_fake: torch.Tensor):
        """
        Calculate the FID score
        images_real: real images with shape (batch_size, channels, height, width)
        images_fake: fake images with shape (batch_size, channels, height, width)
        """
        images_real = images_real.to(self.device)
        images_fake = images_fake.to(self.device)

        if images_real.shape != images_fake.shape:
            raise ValueError(f"Real and fake images must have same dimension ({images_real.shape}) ({images_fake.shape})")

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

        # images are in range [0, 1] but inception model has the following requirements:
        #
        # All pre-trained models expect input images normalized in the same way, i.e. mini-batches of
        # 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 299.
        # The images have to be loaded in to a range of [0, 1] and then normalized
        # using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
        # From: https://pytorch.org/hub/pytorch_vision_inception_v3/

        images_real = self.preprocess(images_real)
        images_fake = self.preprocess(images_fake)

        activation_real = self._get_activations(images_real)
        activation_fake = self._get_activations(images_fake)
        self.activations_real.append(activation_real)
        self.activations_fake.append(activation_fake)

    def calculate(self):
        activations_real = torch.cat(self.activations_real)
        activations_fake = torch.cat(self.activations_fake)

        mu_real, sigma_real = self._get_statistics(activations_real)
        mu_fake, sigma_fake = self._get_statistics(activations_fake)

        fid = self._get_fid(mu_real, sigma_real, mu_fake, sigma_fake)
        return fid.cpu().numpy().item()

    def clear(self):
        self.activations_real = []
        self.activations_fake = []