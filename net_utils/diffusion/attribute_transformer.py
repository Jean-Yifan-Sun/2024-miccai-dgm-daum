import torch.nn as nn
import torch
from typing import Dict, List
import logging
from net_utils.diffusion.attention import SelfAttention


class AttributeTransformer(nn.Module):  # Poor naming, not an actual transformer

    def __init__(self, label_mapping: List[Dict[str, int]], max_labels: int, context_dim: int, heads: int = 1,
                 layers: int = 1, **config):
        super().__init__()
        if max_labels > 0 and label_mapping is None:
            raise ValueError("For max_labels > 0 a label_mapping must be provided.")

        self.label_mapping = {list(e.keys())[0]: list(e.values())[0] for e in label_mapping}
        print("label_mapping: ", self.label_mapping)
        logging.info(f'Label mapping: {self.label_mapping}')
        self.max_labels = max_labels
        if self.max_labels <= 0:
            # Only one embedding linear layer
            self.embeddings = nn.Linear(1, context_dim)
        else:
            # One embedding layer per label
            self.embeddings = torch.nn.ModuleList([nn.Linear(1, context_dim) for _ in range(max_labels)])
        self.attentions = nn.ModuleList([SelfAttention(context_dim, heads=heads, head_dim=context_dim, **config) for _ in range(layers)])
        self.activation = nn.SELU()

    def forward(self, labels: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if len(labels.shape) != 2:
            raise ValueError("AttributeTransformer only supports 2D tensors. Got {}".format(labels.shape))
        B, C = labels.shape

        if self.max_labels > 0:
            if C != len(self.label_mapping):
                raise ValueError("Labels tensor must have the same number of elements as the label_mapping.")

            if C > self.max_labels:
                raise ValueError(f"Labels tensor can't have more than {self.max_labels} labels. Got {C}.")

            # Select the right embedding layer for each label according to label order and apply to column
            context = torch.stack([self.embeddings[e](labels[:, i].unsqueeze(1))
                                   for i, e in enumerate(self.label_mapping.values())], dim=1)

            context = self.activation(context)
        else:
            # reshape context to B*C x 1
            context = labels.reshape(-1, 1)
            # embed each element in the context to a vector of dimension A
            context = self.embedding(context)
            # reshape context to B x C x A
            context = context.reshape(B, C, -1)

        # pass through attention layers
        for attention in self.attentions:
            context = attention(context, mask=mask)

        return context
