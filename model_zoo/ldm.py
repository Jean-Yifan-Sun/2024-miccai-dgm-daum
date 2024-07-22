import random
from enum import Enum, auto
import logging

from dl_utils.config_utils import import_module
from typing import Dict, List, Tuple

import torch
from torch import nn

from net_utils.diffusion.attention import Attention
from net_utils.diffusion.attribute_transformer import AttributeTransformer
from net_utils.diffusion.scheduler import DDPMScheduler
from net_utils.diffusion.denoising_unet import DenoisingUnet
from net_utils.diffusion.transformer import MLP, BasicSpatialTransformer


class LatentDiffusion(nn.Module):
    def __init__(self,
                 T: int,
                 beta_min: float = 1e-4,
                 beta_max: float = 2e-2,
                 first_stage_model = None,
                 guidance: float = 0,
                 **config):
        super().__init__()
        self.T = T
        self.guidance = guidance
        self.input_size = config['input_size']
        self.input_dim = config['input_dim']

        if first_stage_model is None:
            self.first_stage_model = nn.Identity()
            self.has_first_stage = False
            self.diffusion_size = self.input_size
        else:
            model_class = import_module(first_stage_model['module_name'], first_stage_model['class_name'])
            self.first_stage_model = model_class(**(first_stage_model['params']))
            checkpoint = torch.load(first_stage_model['weights'], map_location='cpu')
            self.first_stage_model.load_state_dict(checkpoint['model_weights'])
            self.has_first_stage = True
            compression_factor = 2**len(first_stage_model['params']['layer_multiplier'])
            self.diffusion_size = list(self.input_size)
            self.diffusion_size[-1] = self.diffusion_size[-1] // compression_factor
            self.diffusion_size[-2] = self.diffusion_size[-2] // compression_factor
            self.diffusion_size = tuple(self.diffusion_size)

        self.denoising_unet = DenoisingUnet(**config)

        self.attribute_transformer = AttributeTransformer(**config) if config['context_dim'] > 0 else None

        self.scheduler = DDPMScheduler(T=T, beta_min=float(beta_min), beta_max=float(beta_max))

        self.register_buffer('latent_std_estimate', torch.tensor(1.0))

    def forward(self, x: torch.Tensor, context: torch.Tensor = None, mask: torch.Tensor = None)\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = torch.randint(0, self.T, (x.shape[0],), device=x.device)
        x_t, epsilon = self.diffusion_forward(x, t)

        if context is not None and self.attribute_transformer is not None:
            context = self.attribute_transformer(context.float(), mask)
        else:
            context = None

        epsilon_pred = self.denoising_unet(x_t, t, context, mask)

        return epsilon_pred, epsilon, t

    def diffusion_forward(self, x: torch.Tensor, t: torch.Tensor):
        if self.has_first_stage:
            latent_x = self.first_stage_model.encode(x)
            latent_x = latent_x / self.latent_std_estimate
        else:
            latent_x = x
            latent_x = latent_x.unsqueeze(1)
        x_t, epsilon = self.scheduler.make_noisy(latent_x, t)
        return x_t, epsilon


    def reverse(self, x_t: torch.Tensor, T: int, context: torch.Tensor = None, mask: torch.Tensor = None,
                diffusion_steps: bool = False, diffusion_errors: bool = False) -> torch.Tensor:
        N = x_t.shape[0]
        steps = []
        for t in range(T - 1, -1, -1):
            timestep = torch.tensor([t]).repeat_interleave(N, dim=0).long().to(x_t.device)
            error = self.denoising_unet(x_t, timestep, context, mask)
            if self.guidance > 0:
                unconditional_error = self.denoising_unet(x_t, timestep, None, mask)
                error = (1 + self.guidance) * error - self.guidance * unconditional_error
            if diffusion_errors:
                steps.append(error)
                continue
            x_t = self.scheduler.reverse(x_t, error, timestep)
            if diffusion_steps:
                steps.append(x_t)
        if diffusion_steps or diffusion_errors:
            return steps[::-1]
        return x_t

    def sample(self, N: int, context: torch.Tensor = None, mask: torch.Tensor = None, device: str = 'cuda', latents: bool = True,
               diffusion_steps: bool = False, diffusion_errors: bool = False) -> torch.Tensor:
        if context is not None and self.attribute_transformer is not None:
            context = self.attribute_transformer(context.float(), mask)
        else:
            context = None
        noise_shape = self.diffusion_size

        x_t = torch.randn(N, *noise_shape).to(device)

        T = self.T
        x_t = self.reverse(x_t, T, context, mask, diffusion_steps, diffusion_errors)

        if diffusion_steps or diffusion_errors:
            return x_t

        if not latents and self.has_first_stage:
            x_t = self.first_stage_model.decode(x_t * self.latent_std_estimate)
        if self.input_dim == 3:
            x_t = x_t.squeeze(1)
        return x_t

    def calculate_latent_std(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_first_stage:
            latent_x = self.first_stage_model.encode(x)
        else:
            latent_x = x

        return latent_x.std()

    def set_latent_std(self, latent_std: torch.Tensor):
        self.latent_std_estimate = latent_std

    def finetune(self, flag: bool):
        if not flag:
            for param in self.denoising_unet.parameters():
                param.requires_grad = True
        else:
            attention_params = set()
            for module in self.denoising_unet.modules():
                if isinstance(module, Attention) or isinstance(module, BasicSpatialTransformer):
                    attention_params.update(set(module.parameters()))

            for param in self.denoising_unet.parameters():
                if param not in attention_params:
                    param.requires_grad = False

    def get_label_index(self, label_name):
        if self.attribute_transformer is not None:
            return self.attribute_transformer.label_mapping[label_name]
        else:
            return None
