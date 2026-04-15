from __future__ import annotations

import torch

from helmas3n.src.losses.attention_loss import attention_output_consistency_loss


def test_attention_output_consistency_uses_key_geometry_when_sequence_len_gt_one() -> None:
    query = torch.tensor([[[1.0, 0.0]]])  # [B=1, H=1, D=2]

    pred_k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])  # [1,1,S=2,2]
    pred_v = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])

    # Swap key order so query should attend differently.
    target_k = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    target_v = pred_v.clone()

    losses = attention_output_consistency_loss(
        query=query,
        pred_k=pred_k,
        pred_v=pred_v,
        target_k=target_k,
        target_v=target_v,
    )
    assert losses["total"].item() > 0.01
