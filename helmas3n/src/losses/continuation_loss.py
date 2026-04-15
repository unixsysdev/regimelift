from __future__ import annotations

import torch
import torch.nn.functional as F


def continuation_cross_entropy(
    pred_logits: torch.Tensor,
    target_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy over resumed continuation tokens.

    pred_logits: [B, T, V]
    target_token_ids: [B, T]
    """
    b, t, v = pred_logits.shape
    return F.cross_entropy(pred_logits.reshape(b * t, v), target_token_ids.reshape(b * t))
