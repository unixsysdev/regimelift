from __future__ import annotations

import torch
import torch.nn.functional as F


def mean_cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(x, y, dim=-1).mean()


def mean_squared_error(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x, y)


def top1_agreement(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    a = logits_a.argmax(dim=-1)
    b = logits_b.argmax(dim=-1)
    return (a == b).float().mean()


def sparse_top1_agreement(
    indices_a: torch.Tensor,
    values_a: torch.Tensor,
    indices_b: torch.Tensor,
    values_b: torch.Tensor,
) -> torch.Tensor:
    top_a = indices_a.gather(-1, values_a.argmax(dim=-1, keepdim=True)).squeeze(-1)
    top_b = indices_b.gather(-1, values_b.argmax(dim=-1, keepdim=True)).squeeze(-1)
    return (top_a == top_b).float().mean()


def continuation_match_rate(tokens_a: torch.Tensor, tokens_b: torch.Tensor) -> torch.Tensor:
    # tokens: [B, T]
    return (tokens_a == tokens_b).float().mean()


def long_horizon_drift(tokens_ref: torch.Tensor, tokens_test: torch.Tensor) -> torch.Tensor:
    """Fraction of positions that differ from reference continuation."""
    return 1.0 - continuation_match_rate(tokens_ref, tokens_test)
