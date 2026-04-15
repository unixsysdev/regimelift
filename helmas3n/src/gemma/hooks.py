from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable

import torch


def register_residual_patch_hooks(
    layers,
    patch_map: Dict[int, torch.Tensor],
):
    """Patch per-layer hidden output for prefill handoff experiments.

    patch_map[layer] can be either:
    - [hidden_dim]: applied to last token
    - [seq_len, hidden_dim]: applied to full sequence
    """
    handles = []

    for layer_idx, layer in enumerate(layers):
        if layer_idx not in patch_map:
            continue

        patch = patch_map[layer_idx]

        def _hook(module, inputs, output, idx=layer_idx, p=patch):
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None

            patched = hidden.clone()
            target = patched
            # Gemma 3n can emit [altup, batch, seq, hidden]; patch active stream at index 0.
            if patched.ndim == 4:
                target = patched[0]
            if p.ndim == 1:
                target[:, -1, :] = p.to(hidden.device, hidden.dtype)
            elif p.ndim == 2:
                seq = min(p.shape[0], target.shape[1])
                target[:, :seq, :] = p[:seq].to(hidden.device, hidden.dtype)
            else:
                raise ValueError(f"Unsupported patch rank for layer {idx}: {p.ndim}")

            if rest is None:
                return patched
            return (patched, *rest)

        handles.append(layer.register_forward_hook(_hook))

    return handles


def clear_hooks(handles: Iterable) -> None:
    for handle in handles:
        handle.remove()


@contextmanager
def residual_patch_context(layers, patch_map: Dict[int, torch.Tensor]):
    handles = register_residual_patch_hooks(layers, patch_map)
    try:
        yield
    finally:
        clear_hooks(handles)
