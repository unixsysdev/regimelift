from __future__ import annotations

import torch
from torch import nn


class ConditionalFlowRefinement(nn.Module):
    """Phase-2 placeholder: refine a coarse uplift with a learned flow field."""

    def __init__(self, hidden_dim: int, cond_dim: int = 0, flow_hidden_dim: int = 1024) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim

        in_dim = hidden_dim + cond_dim + 1  # + time scalar
        self.velocity = nn.Sequential(
            nn.Linear(in_dim, flow_hidden_dim),
            nn.SiLU(),
            nn.Linear(flow_hidden_dim, flow_hidden_dim),
            nn.SiLU(),
            nn.Linear(flow_hidden_dim, hidden_dim),
        )

    def forward(
        self,
        coarse_state: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
        step_size: float = 1.0,
    ) -> torch.Tensor:
        if cond is None:
            cond = torch.zeros(coarse_state.size(0), 0, device=coarse_state.device, dtype=coarse_state.dtype)
        if t.ndim == 1:
            t = t.unsqueeze(-1)

        x = torch.cat([coarse_state, cond, t], dim=-1)
        v = self.velocity(x)
        return coarse_state + step_size * v


class LayerConditionedFlowUplift(nn.Module):
    """Flow wrapper conditioned on normalized layer index."""

    def __init__(self, hidden_dim: int, num_layers: int, flow_hidden_dim: int = 1024) -> None:
        super().__init__()
        self.num_layers = max(num_layers, 1)
        self.flow = ConditionalFlowRefinement(hidden_dim=hidden_dim, cond_dim=1, flow_hidden_dim=flow_hidden_dim)

    def forward(self, z: torch.Tensor, layer_ids: torch.Tensor) -> torch.Tensor:
        cond = layer_ids.float().unsqueeze(-1) / float(self.num_layers - 1 if self.num_layers > 1 else 1)
        t = torch.ones(z.size(0), 1, device=z.device, dtype=z.dtype)
        return self.flow(coarse_state=z, t=t, cond=cond)
