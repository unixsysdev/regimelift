from __future__ import annotations

import torch

from helmas3n.src.eval.handoff_metrics import continuation_match_rate, long_horizon_drift, top1_agreement


def test_top1_agreement() -> None:
    a = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    b = torch.tensor([[0.5, 3.0], [1.0, 2.0]])
    score = top1_agreement(a, b)
    assert torch.isclose(score, torch.tensor(0.5))


def test_continuation_and_drift() -> None:
    ref = torch.tensor([[1, 2, 3, 4]])
    test = torch.tensor([[1, 2, 0, 4]])
    match = continuation_match_rate(ref, test)
    drift = long_horizon_drift(ref, test)
    assert torch.isclose(match, torch.tensor(0.75))
    assert torch.isclose(drift, torch.tensor(0.25))
