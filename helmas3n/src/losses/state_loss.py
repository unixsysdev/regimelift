from __future__ import annotations

import torch
import torch.nn.functional as F


def state_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mse_weight: float = 1.0,
    cosine_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    mse = F.mse_loss(pred, target)
    cosine = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
    total = mse_weight * mse + cosine_weight * cosine
    return {
        "total": total,
        "mse": mse,
        "cosine": cosine,
    }
