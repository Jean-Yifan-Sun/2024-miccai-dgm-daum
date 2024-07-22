from typing import Tuple, Any

import torch
from torch import Tensor


class DDPMScheduler:
    def __init__(self, T: int = 1000, beta_min: float = 1e-4, beta_max: float = 2e-2):
        self.T = T
        self.beta_min = beta_min
        self.beta_max = beta_max

        self.betas = torch.linspace(beta_min, beta_max, T) #.to('cuda')
        self.sqrt_betas = torch.sqrt(self.betas)

        self.alphas = 1 - self.betas
        alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_bar = torch.sqrt(alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1 - alphas_bar)

    def make_noisy(self, x_0: torch.Tensor, t: torch.Tensor) -> Tuple[Tensor, Tensor]:
        """Adds Gaussian noise scaled by sqrt(1 - beta_t) to x_0.
        t contains random timesteps for the different samples."""
        t = t.cpu().long()
        epsilon = torch.randn_like(x_0).to(x_0.device)

        expander = [1] * (len(x_0.shape) - 1)
        sqrt_alpha_bar = self.sqrt_alphas_bar[t].view(-1, *expander).to(x_0.device)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_bar[t].view(-1, *expander).to(x_0.device)
        return (x_0 * sqrt_alpha_bar + epsilon * sqrt_one_minus_alpha_bar), epsilon

    def reverse(self, x_t: torch.Tensor, predicted_epsilon: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predicts x_t-1 from x_t by reversing the diffusion process.
        t contains the same time steps for all samples."""
        assert torch.all(t == t[0]), "timesteps must be the same for all samples"
        t = t.cpu()
        if t[0] > 1:
            z = torch.randn_like(x_t).to(x_t.device)
        else:
            z = torch.zeros_like(x_t).to(x_t.device)

        expander = [1] * (len(x_t.shape) - 1)
        inv_sqrt_alpha = torch.sqrt(1 / self.alphas[t]).view(-1, *expander).to(x_t.device)
        noise_term = (self.betas[t] / self.sqrt_one_minus_alphas_bar[t]).view(-1, *expander).to(x_t.device)
        sqrt_beta = self.sqrt_betas[t].view(-1, *expander).to(x_t.device)

        return (inv_sqrt_alpha * (x_t - predicted_epsilon * noise_term) + (z * sqrt_beta))