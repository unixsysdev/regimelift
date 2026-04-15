from __future__ import annotations

import torch

from helmas3n.src.models.uplift_baselines import (
    GlobalLinearUplift,
    IdentityUplift,
    LayerwiseMeanDeltaUplift,
)
from helmas3n.src.models.uplift_linear import LayerwiseLinearUplift
from helmas3n.src.models.uplift_lowrank import LayerwiseLowRankUplift
from helmas3n.src.models.uplift_mlp import LayerwiseMLPUplift


def test_linear_uplift_shape_and_identity_init() -> None:
    model = LayerwiseLinearUplift(num_layers=4, hidden_dim=8)
    z = torch.randn(6, 8)
    layer = torch.tensor([0, 1, 2, 3, 1, 0])
    out = model(z, layer)
    assert out.shape == z.shape
    assert torch.allclose(out, z, atol=1e-6)


def test_lowrank_uplift_shape() -> None:
    model = LayerwiseLowRankUplift(num_layers=4, hidden_dim=8, rank=3)
    z = torch.randn(6, 8)
    layer = torch.tensor([0, 1, 2, 3, 1, 0])
    out = model(z, layer)
    assert out.shape == z.shape


def test_mlp_uplift_shape_shared_and_unshared() -> None:
    z = torch.randn(5, 10)
    layer = torch.tensor([0, 1, 0, 2, 1])

    shared = LayerwiseMLPUplift(num_layers=3, hidden_dim=10, shared=True)
    out_shared = shared(z, layer)
    assert out_shared.shape == z.shape

    unshared = LayerwiseMLPUplift(num_layers=3, hidden_dim=10, shared=False)
    out_unshared = unshared(z, layer)
    assert out_unshared.shape == z.shape


def test_baseline_models_shape() -> None:
    z = torch.randn(4, 12)
    layer = torch.tensor([0, 1, 2, 1])

    identity = IdentityUplift()
    assert torch.allclose(identity(z, layer), z)

    mean_delta = LayerwiseMeanDeltaUplift(num_layers=3, hidden_dim=12)
    out_md = mean_delta(z, layer)
    assert out_md.shape == z.shape

    global_linear = GlobalLinearUplift(hidden_dim=12)
    out_gl = global_linear(z, layer)
    assert out_gl.shape == z.shape
