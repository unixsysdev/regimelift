from __future__ import annotations

import torch
from torch import nn


class LayerwiseLowRankUplift(nn.Module):
    """Per-layer low-rank residual uplift: z_hat = z + U_l V_l^T z."""

    def __init__(self, num_layers: int, hidden_dim: int, rank: int = 64) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.rank = rank

        self.u = nn.Parameter(torch.randn(num_layers, hidden_dim, rank) * 0.01)
        self.v = nn.Parameter(torch.randn(num_layers, hidden_dim, rank) * 0.01)

    def forward(self, z: torch.Tensor, layer_ids: torch.Tensor) -> torch.Tensor:
        u = self.u[layer_ids]
        v = self.v[layer_ids]
        proj = torch.bmm(v.transpose(1, 2), z.unsqueeze(-1)).squeeze(-1)
        delta = torch.bmm(u, proj.unsqueeze(-1)).squeeze(-1)
        return z + delta
