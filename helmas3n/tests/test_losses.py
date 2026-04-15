from __future__ import annotations

import torch

from helmas3n.src.losses.logit_loss import project_topk_logits, sparse_topk_kl_loss
from helmas3n.src.losses.state_loss import state_reconstruction_loss


def test_state_reconstruction_loss_zero_on_match() -> None:
    x = torch.randn(4, 16)
    losses = state_reconstruction_loss(x, x, mse_weight=1.0, cosine_weight=0.5)
    assert losses["mse"].abs().item() < 1e-7
    assert losses["cosine"].abs().item() < 1e-6


def test_sparse_topk_kl_non_negative() -> None:
    pred = torch.tensor([[2.0, 1.0, 0.0]])
    target = torch.tensor([[1.5, 0.5, -0.5]])
    kl = sparse_topk_kl_loss(pred, target)
    assert kl.item() >= -1e-6


def test_project_topk_logits_shape() -> None:
    hidden = torch.randn(2, 6)
    indices = torch.tensor([[0, 2, 4], [1, 3, 5]])
    weight = torch.randn(8, 6)
    logits = project_topk_logits(hidden, indices, lm_head_weight=weight)
    assert logits.shape == (2, 3)
