from __future__ import annotations

import torch
import torch.nn.functional as F


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x * torch.rsqrt(var + eps)
    return x_norm * weight


def project_topk_logits(
    hidden: torch.Tensor,
    token_indices: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None = None,
    final_norm_weight: torch.Tensor | None = None,
    final_norm_eps: float = 1e-6,
) -> torch.Tensor:
    """Projects hidden states to selected vocab indices without full vocab matmul."""
    if final_norm_weight is not None:
        hidden = rms_norm(hidden, final_norm_weight.to(hidden.device, hidden.dtype), final_norm_eps)

    selected_w = lm_head_weight[token_indices].to(hidden.device, hidden.dtype)  # [B, K, D]
    logits = torch.einsum("bkd,bd->bk", selected_w, hidden)

    if lm_head_bias is not None:
        logits = logits + lm_head_bias[token_indices].to(hidden.device, hidden.dtype)

    return logits


def sparse_topk_kl_loss(
    pred_logits_subset: torch.Tensor,
    target_logits_subset: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL(target || pred) computed on target top-k support."""
    tgt = F.softmax(target_logits_subset / temperature, dim=-1)
    pred_log = F.log_softmax(pred_logits_subset / temperature, dim=-1)
    return F.kl_div(pred_log, tgt, reduction="batchmean") * (temperature**2)
