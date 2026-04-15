from __future__ import annotations

import torch
from torch import nn


class IdentityUplift(nn.Module):
    """Null baseline: z_hat = z."""

    def forward(self, z: torch.Tensor, layer_ids: torch.Tensor) -> torch.Tensor:
        del layer_ids
        return z


class LayerwiseMeanDeltaUplift(nn.Module):
    """Per-layer additive baseline: z_hat = z + delta_l."""

    def __init__(self, num_layers: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.delta = nn.Parameter(torch.zeros(num_layers, hidden_dim))

    def set_delta(self, delta: torch.Tensor) -> None:
        if delta.shape != self.delta.shape:
            raise ValueError(f"delta shape mismatch: expected {self.delta.shape}, got {delta.shape}")
        with torch.no_grad():
            self.delta.copy_(delta)

    def forward(self, z: torch.Tensor, layer_ids: torch.Tensor) -> torch.Tensor:
        return z + self.delta[layer_ids]


class GlobalLinearUplift(nn.Module):
    """Shared affine map across layers: z_hat = Wz + b."""

    def __init__(self, hidden_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(hidden_dim))
            if self.proj.bias is not None:
                self.proj.bias.zero_()

    def forward(self, z: torch.Tensor, layer_ids: torch.Tensor) -> torch.Tensor:
        del layer_ids
        return self.proj(z)
