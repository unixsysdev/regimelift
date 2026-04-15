from __future__ import annotations

from typing import Dict

import torch

from helmas3n.src.gemma.hooks import residual_patch_context


def greedy_decode_from_past(
    runner,
    past_key_values,
    last_token_ids: torch.Tensor,
    regime: str,
    max_new_tokens: int,
    seed_token_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Greedy continuation with an existing cache."""
    generated = []
    current = last_token_ids
    steps = max_new_tokens
    if seed_token_ids is not None:
        generated.append(seed_token_ids)
        current = seed_token_ids
        steps = max(max_new_tokens - 1, 0)
    regime_cfg = runner.regimes.get(regime, {})
    runtime_overrides = regime_cfg.get("runtime_overrides", {})

    cache_layers = getattr(past_key_values, "layers", None)
    expected_layers = runner.expected_cache_layer_count(regime)
    if isinstance(past_key_values, (tuple, list)):
        past_key_values = runner.to_dynamic_cache_for_regime(past_key_values, regime)
    elif cache_layers is not None and len(cache_layers) != expected_layers:
        # Cross-regime handoff can produce compact DynamicCache; rebuild with the decode regime layout.
        past_key_values = runner.to_dynamic_cache_for_regime(past_key_values, regime)

    with (
        torch.no_grad(),
        runner._apply_model_overrides(regime_cfg.get("model_overrides", {})),
        runner._apply_text_runtime_overrides(regime_cfg.get("text_overrides", {})),
    ):
        for _ in range(steps):
            outputs = runner.model(
                input_ids=current,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                **runtime_overrides,
            )
            logits = outputs.logits[:, -1, :]
            next_tok = logits.argmax(dim=-1, keepdim=True)
            generated.append(next_tok)
            current = next_tok
            past_key_values = outputs.past_key_values

    return torch.cat(generated, dim=1) if generated else torch.empty(0)


def resume_with_residual_patch(
    runner,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    patch_map: Dict[int, torch.Tensor],
    regime: str,
    max_new_tokens: int,
    decode_regime: str | None = None,
) -> torch.Tensor:
    """Patch residuals during prefill, then continue decoding from resulting cache."""
    layers = runner.get_transformer_layers()
    regime_cfg = runner.regimes.get(regime, {})
    runtime_overrides = regime_cfg.get("runtime_overrides", {})
    decode_regime = decode_regime or regime

    with (
        torch.no_grad(),
        runner._apply_model_overrides(regime_cfg.get("model_overrides", {})),
        runner._apply_text_runtime_overrides(regime_cfg.get("text_overrides", {})),
    ):
        with residual_patch_context(layers, patch_map):
            outputs = runner.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
                **runtime_overrides,
            )

    last_token = input_ids[:, -1:]
    first_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return greedy_decode_from_past(
        runner=runner,
        past_key_values=outputs.past_key_values,
        last_token_ids=last_token,
        regime=decode_regime,
        max_new_tokens=max_new_tokens,
        seed_token_ids=first_token,
    )


def resume_with_kv_cache(
    runner,
    uplifted_past_key_values,
    last_token_ids: torch.Tensor,
    regime: str,
    max_new_tokens: int,
    seed_token_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    return greedy_decode_from_past(
        runner=runner,
        past_key_values=uplifted_past_key_values,
        last_token_ids=last_token_ids,
        regime=regime,
        max_new_tokens=max_new_tokens,
        seed_token_ids=seed_token_ids,
    )
