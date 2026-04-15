from __future__ import annotations

import torch
from torch import nn


class _ResidualMLP(nn.Module):
    def __init__(self, hidden_dim: int, expansion: float = 2.0) -> None:
        super().__init__()
        inner = int(hidden_dim * expansion)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.GELU(),
            nn.Linear(inner, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class LayerwiseMLPUplift(nn.Module):
    """Per-layer or shared residual MLP uplift."""

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        expansion: float = 2.0,
        shared: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.shared = shared

        if shared:
            self.shared_mlp = _ResidualMLP(hidden_dim, expansion)
            self.layer_mlps = None
        else:
            self.shared_mlp = None
            self.layer_mlps = nn.ModuleList(
                [_ResidualMLP(hidden_dim, expansion) for _ in range(num_layers)]
            )

    def forward(self, z: torch.Tensor, layer_ids: torch.Tensor) -> torch.Tensor:
        if self.shared:
            return self.shared_mlp(z)

        order = torch.argsort(layer_ids)
        z_sorted = z.index_select(0, order)
        layer_sorted = layer_ids.index_select(0, order)

        out_sorted = torch.empty_like(z_sorted)
        unique_layers, counts = torch.unique_consecutive(layer_sorted, return_counts=True)

        start = 0
        for layer_idx, count in zip(unique_layers.tolist(), counts.tolist()):
            end = start + count
            out_sorted[start:end] = self.layer_mlps[layer_idx](z_sorted[start:end])
            start = end

        out = torch.empty_like(out_sorted)
        out[order] = out_sorted
        return out
