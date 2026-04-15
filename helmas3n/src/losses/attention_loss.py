from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _attend(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    # query: [B, H, D], key/value: [B, H, S, D]
    scores = torch.einsum("bhd,bhsd->bhs", query, key) / math.sqrt(query.size(-1))
    probs = scores.softmax(dim=-1)
    out = torch.einsum("bhs,bhsd->bhd", probs, value)
    return out


def attention_output_consistency_loss(
    query: torch.Tensor,
    pred_k: torch.Tensor,
    pred_v: torch.Tensor,
    target_k: torch.Tensor,
    target_v: torch.Tensor,
    mse_weight: float = 1.0,
    cosine_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    pred_out = _attend(query, pred_k, pred_v)
    target_out = _attend(query, target_k, target_v)

    mse = F.mse_loss(pred_out, target_out)
    cosine = 1.0 - F.cosine_similarity(
        pred_out.reshape(pred_out.size(0), -1),
        target_out.reshape(target_out.size(0), -1),
        dim=-1,
    ).mean()
    total = mse_weight * mse + cosine_weight * cosine
    return {
        "total": total,
        "mse": mse,
        "cosine": cosine,
    }


def direct_kv_consistency_loss(
    pred_k: torch.Tensor,
    pred_v: torch.Tensor,
    target_k: torch.Tensor,
    target_v: torch.Tensor,
    mse_weight: float = 1.0,
    cosine_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    k_mse = F.mse_loss(pred_k, target_k)
    v_mse = F.mse_loss(pred_v, target_v)
    mse = 0.5 * (k_mse + v_mse)

    pred_flat = torch.cat([pred_k.reshape(pred_k.size(0), -1), pred_v.reshape(pred_v.size(0), -1)], dim=-1)
    tgt_flat = torch.cat([target_k.reshape(target_k.size(0), -1), target_v.reshape(target_v.size(0), -1)], dim=-1)
    cosine = 1.0 - F.cosine_similarity(pred_flat, tgt_flat, dim=-1).mean()
    total = mse_weight * mse + cosine_weight * cosine
    return {
        "total": total,
        "mse": mse,
        "cosine": cosine,
        "k_mse": k_mse,
        "v_mse": v_mse,
    }
