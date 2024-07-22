import numpy as np
import torch
import torch.nn.functional as F


class KLDLoss:
    def __init__(self, gamma=10.0, max_capacity=25):
        super(KLDLoss, self).__init__()
        self.num_iter = 0
        self.max_capacity = max_capacity
        self.capacity_max_iter = 1e5
        self.gamma = gamma

    def __call__(self, dist):
        self.num_iter += 1
        mu = dist['z_mu']
        log_var = dist['z_logvar']

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        c = 0
        if self.max_capacity > 0:
            c = np.clip(self.max_capacity / self.capacity_max_iter * self.num_iter, 0, self.max_capacity)
        loss = self.gamma * (kld_loss - c).abs()


        return loss

class AttributeRegularizationLoss:

    def __init__(self, gamma=1, factor=1):
        super(AttributeRegularizationLoss, self).__init__()
        self.gamma = gamma
        self.factor = factor

    def __call__(self, z, labels, reg_dims):
        """
        Computes the attribute regularization loss for the given latent space
        :param z: Latent representation -- torch Variable, (B, N)
        :param labels: Labels to be used for regularization -- torch Variable, (B, L)
        :param reg_dims: Indices of the latent dimensions to be regularized -- set (L, )
        :return: Regularization loss, torch Variable
        """
        # compute latent distance matrix
        lc_dist_mat = self.pairwise_distance_matrix(z[:, reg_dims])

        # compute attribute distance matrix
        attribute_dist_mat = self.pairwise_distance_matrix(labels)

        lc_tanh = torch.tanh(lc_dist_mat * self.factor)
        attribute_sign = torch.sign(attribute_dist_mat)
        sign_loss = F.l1_loss(lc_tanh, attribute_sign.float(), reduction='mean')
        return self.gamma * sign_loss * len(reg_dims)


    @staticmethod
    def pairwise_distance_matrix(x):
        assert len(x.shape) == 2, "Input tensor must be 2D (B, L)"
        B, L = x.shape

        # Expand and transpose x to get two tensors of shape (B, B, L)
        x1 = x.unsqueeze(1).expand(B, B, L)
        x2 = x.unsqueeze(0).expand(B, B, L)

        # Compute the pairwise distance for each column
        # Resulting shape will be (B, B, L), then transpose to (L, B, B)
        distance_matrix = (x1 - x2).transpose(0, 2)

        return distance_matrix

