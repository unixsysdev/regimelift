from __future__ import annotations

from types import SimpleNamespace

import torch

from helmas3n.src.gemma.runner import GemmaRunner


def _kv_layer(seed: float) -> SimpleNamespace:
    keys = torch.full((1, 1, 4, 2), seed, dtype=torch.float32)
    values = torch.full((1, 1, 4, 2), seed + 1.0, dtype=torch.float32)
    return SimpleNamespace(keys=keys, values=values)


def _build_runner(layer_types: list[str]) -> GemmaRunner:
    runner = GemmaRunner.__new__(GemmaRunner)
    runner.model = SimpleNamespace(config=SimpleNamespace(layer_types=layer_types))
    return runner


def test_get_layer_kv_resolves_shared_anchor_from_shared_layers() -> None:
    runner = _build_runner(["s", "f", "s", "s"])
    base = _kv_layer(0.0)
    replacement = _kv_layer(10.0)
    past = SimpleNamespace(layers=[base, _kv_layer(1.0)], shared_layers={0: (replacement.keys, replacement.values)})

    k, v = runner.get_layer_kv(past, layer_idx=3)
    assert torch.allclose(k, replacement.keys)
    assert torch.allclose(v, replacement.values)


def test_get_layer_kv_at_position_maps_absolute_to_cache_position() -> None:
    runner = _build_runner(["s", "f", "s", "f"])
    layer = _kv_layer(0.0)
    # Encode position index in key tensor for easy assertion.
    for i in range(layer.keys.size(2)):
        layer.keys[0, 0, i, 0] = float(i)
    past = SimpleNamespace(layers=[layer, _kv_layer(1.0)], shared_layers={})

    # sequence_length=10 and cache_len=4 => cached span is [6, 9], token_position=8 maps to index 2.
    k, _ = runner.get_layer_kv_at_position(past, layer_idx=0, token_position=8, sequence_length=10)
    assert torch.isclose(k[0, 0], torch.tensor(2.0))
