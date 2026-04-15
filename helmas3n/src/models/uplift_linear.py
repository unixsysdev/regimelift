from __future__ import annotations

import torch
from torch import nn


class LayerwiseLinearUplift(nn.Module):
    """Per-layer affine transport: z_hat = W_l z + b_l."""

    def __init__(self, num_layers: int, hidden_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        eye = torch.eye(hidden_dim).unsqueeze(0).repeat(num_layers, 1, 1)
        self.weight = nn.Parameter(eye)
        self.bias = nn.Parameter(torch.zeros(num_layers, hidden_dim)) if bias else None

    def forward(self, z: torch.Tensor, layer_ids: torch.Tensor) -> torch.Tensor:
        w = self.weight[layer_ids]
        out = torch.bmm(w, z.unsqueeze(-1)).squeeze(-1)
        if self.bias is not None:
            out = out + self.bias[layer_ids]
        return out
