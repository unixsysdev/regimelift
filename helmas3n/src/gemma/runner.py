from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import copy
from pathlib import Path
from typing import Any, Dict

import torch


DTYPE_MAP: Dict[str, torch.dtype] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


@dataclass
class ForwardRun:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...]
    logits: torch.Tensor | None
    past_key_values: Any


class GemmaRunner:
    """Regime-aware runner for paired low/full forwards in one checkpoint."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        regimes: dict[str, dict[str, Any]] | None = None,
        model_load_overrides: dict[str, Any] | None = None,
        text_only: bool = True,
    ) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        torch_dtype = DTYPE_MAP.get(dtype.lower(), torch.bfloat16)
        self.device = torch.device(device)
        self.regimes = regimes or {}
        self.model_load_overrides = model_load_overrides or {}
        self.text_only = text_only

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if self.text_only:
            base_cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
            text_cfg = getattr(base_cfg, "text_config", base_cfg)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=text_cfg,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                **self.model_load_overrides,
            ).to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                **self.model_load_overrides,
            ).to(self.device)
        self.model.eval()

    def get_num_layers(self) -> int:
        layers = self.get_transformer_layers()
        return len(layers)

    def get_hidden_dim(self) -> int:
        text_cfg = self._get_text_config()
        if hasattr(text_cfg, "hidden_size"):
            return int(text_cfg.hidden_size)
        if hasattr(text_cfg, "n_embd"):
            return int(text_cfg.n_embd)
        raise AttributeError("Could not infer hidden dimension from model config")

    def get_num_attention_heads(self) -> int:
        text_cfg = self._get_text_config()
        if hasattr(text_cfg, "num_attention_heads"):
            return int(text_cfg.num_attention_heads)
        if hasattr(text_cfg, "n_head"):
            return int(text_cfg.n_head)
        raise AttributeError("Could not infer number of attention heads from model config")

    def get_num_key_value_heads(self) -> int:
        text_cfg = self._get_text_config()
        if hasattr(text_cfg, "num_key_value_heads"):
            return int(text_cfg.num_key_value_heads)
        if hasattr(text_cfg, "num_attention_heads"):
            return int(text_cfg.num_attention_heads)
        raise AttributeError("Could not infer number of KV heads from model config")

    def get_head_dim(self) -> int:
        text_cfg = self._get_text_config()
        if hasattr(text_cfg, "head_dim"):
            return int(text_cfg.head_dim)
        hidden = self.get_hidden_dim()
        heads = self.get_num_attention_heads()
        return hidden // heads

    def tokenize(self, prompt: str, max_length: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        return input_ids, attention_mask

    def forward_prefix(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        regime: str,
        capture_kv: bool,
        capture_logits: bool,
        output_attentions: bool = False,
    ) -> ForwardRun:
        regime_cfg = self.regimes.get(regime, {})
        runtime_overrides = regime_cfg.get("runtime_overrides", {})

        with (
            torch.no_grad(),
            self._apply_model_overrides(regime_cfg.get("model_overrides", {})),
            self._apply_text_runtime_overrides(regime_cfg.get("text_overrides", {})),
        ):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=output_attentions,
                use_cache=capture_kv,
                return_dict=True,
                **runtime_overrides,
            )

        logits = outputs.logits if capture_logits else None
        past_key_values = outputs.past_key_values if capture_kv else None
        return ForwardRun(
            input_ids=input_ids,
            attention_mask=attention_mask,
            hidden_states=outputs.hidden_states,
            logits=logits,
            past_key_values=past_key_values,
        )

    def describe_regime(self, regime: str) -> Dict[str, Any]:
        regime_cfg = self.regimes.get(regime, {})
        with (
            self._apply_model_overrides(regime_cfg.get("model_overrides", {})),
            self._apply_text_runtime_overrides(regime_cfg.get("text_overrides", {})),
        ):
            text_cfg = self._get_text_config()
            layers = self.get_transformer_layers()
            effective = {
                "altup_active_idx": int(getattr(text_cfg, "altup_active_idx", 0)),
                "altup_correct_scale": bool(getattr(text_cfg, "altup_correct_scale", True)),
                "num_kv_shared_layers": int(getattr(text_cfg, "num_kv_shared_layers", 0)),
                "activation_sparsity_pattern": [
                    float(getattr(layer.mlp, "activation_sparsity", 0.0)) for layer in layers
                ],
                "kv_shared_layer_flags": [
                    bool(getattr(layer.self_attn, "is_kv_shared_layer", False)) for layer in layers
                ],
            }

        report: Dict[str, Any] = {
            "regime": regime,
            "model_overrides": copy.deepcopy(regime_cfg.get("model_overrides", {})),
            "runtime_overrides": copy.deepcopy(regime_cfg.get("runtime_overrides", {})),
            "text_overrides": copy.deepcopy(regime_cfg.get("text_overrides", {})),
            "effective": effective,
        }
        return report

    def export_logit_head_snapshot(self, out_path: str | Path) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        lm_head = self.model.get_output_embeddings()
        payload: Dict[str, Any] = {
            "lm_head_weight": lm_head.weight.detach().cpu(),
            "lm_head_bias": None,
            "final_norm_weight": None,
            "final_norm_bias": None,
            "final_norm_eps": 1e-6,
        }

        if getattr(lm_head, "bias", None) is not None:
            payload["lm_head_bias"] = lm_head.bias.detach().cpu()

        final_norm = self.get_final_norm_module()
        if final_norm is not None:
            if getattr(final_norm, "weight", None) is not None:
                payload["final_norm_weight"] = final_norm.weight.detach().cpu()
            if getattr(final_norm, "bias", None) is not None:
                payload["final_norm_bias"] = final_norm.bias.detach().cpu()
            if hasattr(final_norm, "eps"):
                payload["final_norm_eps"] = float(final_norm.eps)

        torch.save(payload, out_path)

    def normalize_hidden_state(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Return hidden state as [batch, seq, hidden]."""
        if hidden_state.ndim == 4:
            text_cfg = self._get_text_config()
            active_idx = int(getattr(text_cfg, "altup_active_idx", 0))
            return hidden_state[active_idx]
        if hidden_state.ndim == 3:
            return hidden_state
        raise ValueError(f"Unsupported hidden state rank: {hidden_state.ndim}")

    def get_transformer_layers(self):
        if (
            hasattr(self.model, "model")
            and hasattr(self.model.model, "language_model")
            and hasattr(self.model.model.language_model, "layers")
        ):
            return self.model.model.language_model.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        if hasattr(self.model, "layers"):
            return self.model.layers
        raise AttributeError("Could not locate transformer layer stack on model")

    def get_final_norm_module(self):
        if (
            hasattr(self.model, "model")
            and hasattr(self.model.model, "language_model")
            and hasattr(self.model.model.language_model, "norm")
        ):
            return self.model.model.language_model.norm
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            return self.model.model.norm
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "ln_f"):
            return self.model.transformer.ln_f
        return None

    def to_legacy_past_key_values(self, past_key_values: Any):
        if past_key_values is None:
            return None
        if isinstance(past_key_values, (tuple, list)):
            return tuple(past_key_values)
        if hasattr(past_key_values, "to_legacy_cache"):
            try:
                return past_key_values.to_legacy_cache()
            except Exception:
                pass
        if hasattr(past_key_values, "layers"):
            legacy = []
            for layer_idx in range(self.get_num_layers()):
                key_states, value_states = self.get_layer_kv(past_key_values, layer_idx)
                legacy.append((key_states, value_states))
            return tuple(legacy)
        return past_key_values

    def expected_cache_layer_count(self, regime: str | None = None) -> int:
        num_layers = self.get_num_layers()
        shared = int(getattr(self._get_text_config(), "num_kv_shared_layers", 0))
        if regime is not None:
            regime_cfg = self.regimes.get(regime, {})
            text_overrides = regime_cfg.get("text_overrides", {})
            if "num_kv_shared_layers" in text_overrides:
                shared = int(text_overrides["num_kv_shared_layers"])
        return max(num_layers - max(shared, 0), 1)

    def to_dynamic_cache_for_regime(self, past_key_values: Any, regime: str):
        from transformers.cache_utils import DynamicCache

        if past_key_values is None:
            return None

        legacy = self.to_legacy_past_key_values(past_key_values)
        if not isinstance(legacy, (tuple, list)):
            return legacy

        del regime
        # Avoid config-driven cache layer truncation for Gemma 3n shared-KV setups.
        return DynamicCache(ddp_cache_data=legacy)

    def get_layer_kv(self, past_key_values: Any, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve KV tensors for a logical transformer layer from DynamicCache or legacy tuples."""
        if past_key_values is None:
            raise KeyError("past_key_values is None")

        if isinstance(past_key_values, (tuple, list)):
            if layer_idx < 0 or layer_idx >= len(past_key_values):
                raise KeyError(f"Layer index {layer_idx} out of range for legacy cache of len={len(past_key_values)}")
            return self._extract_kv_pair(past_key_values[layer_idx])

        cache_layers = getattr(past_key_values, "layers", None)
        if cache_layers is not None and layer_idx < len(cache_layers):
            return self._extract_kv_pair(cache_layers[layer_idx])

        # DynamicCache with KV sharing can keep fewer cache slots than model layers.
        anchor_idx = self._infer_shared_anchor_index(past_key_values, layer_idx)
        if anchor_idx is not None:
            shared_layers = getattr(past_key_values, "shared_layers", None)
            if isinstance(shared_layers, dict) and anchor_idx in shared_layers:
                return self._extract_kv_pair(shared_layers[anchor_idx])
            if cache_layers is not None and anchor_idx < len(cache_layers):
                return self._extract_kv_pair(cache_layers[anchor_idx])

        raise KeyError(f"Could not resolve cache tensors for layer {layer_idx}")

    def get_layer_kv_at_position(
        self,
        past_key_values: Any,
        layer_idx: int,
        token_position: int,
        sequence_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-head K/V at a token position, accounting for sliding-window cache truncation."""
        k_layer, v_layer = self.get_layer_kv(past_key_values, layer_idx)
        cache_len = int(k_layer.size(-2))
        if cache_len <= 0:
            raise IndexError(f"Empty KV cache for layer {layer_idx}")

        cache_offset = max(int(sequence_length) - cache_len, 0)
        cache_pos = int(token_position) - cache_offset
        if cache_pos < 0 or cache_pos >= cache_len:
            raise IndexError(
                f"Token position {token_position} is outside cached span "
                f"[{cache_offset}, {cache_offset + cache_len - 1}] for layer {layer_idx}"
            )

        return k_layer[0, :, cache_pos, :], v_layer[0, :, cache_pos, :]

    @staticmethod
    def _extract_kv_pair(layer_obj: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(layer_obj, "keys") and hasattr(layer_obj, "values"):
            return layer_obj.keys, layer_obj.values
        if isinstance(layer_obj, (tuple, list)) and len(layer_obj) >= 2:
            return layer_obj[0], layer_obj[1]
        raise KeyError(f"Could not extract KV pair from cache layer object of type {type(layer_obj)}")

    def _infer_shared_anchor_index(self, past_key_values: Any, layer_idx: int) -> int | None:
        cache_layers = getattr(past_key_values, "layers", None)
        if cache_layers is None:
            return None
        cache_len = len(cache_layers)
        if cache_len == 0:
            return None

        text_cfg = self._get_text_config()
        layer_types = getattr(text_cfg, "layer_types", None)
        if not isinstance(layer_types, (list, tuple)):
            return cache_len - 1
        if layer_idx < 0 or layer_idx >= len(layer_types):
            return None

        target_type = layer_types[layer_idx]
        # Shared layers typically reuse the latest non-shared cache slot for the same layer type.
        for i in range(min(cache_len - 1, layer_idx - 1), -1, -1):
            if layer_types[i] == target_type:
                return i

        # Fallback: use the most recent available cache slot.
        return cache_len - 1

    @contextmanager
    def _apply_model_overrides(self, overrides: dict[str, Any]):
        if not overrides:
            yield
            return

        original: dict[str, Any] = {}
        try:
            for dotted, value in overrides.items():
                obj, attr = self._resolve_parent_attr(self.model.config, dotted)
                key = dotted
                original[key] = getattr(obj, attr)
                setattr(obj, attr, value)
            yield
        finally:
            for dotted, old_value in original.items():
                obj, attr = self._resolve_parent_attr(self.model.config, dotted)
                setattr(obj, attr, old_value)

    @contextmanager
    def _apply_text_runtime_overrides(self, overrides: dict[str, Any]):
        if not overrides:
            yield
            return

        layers = self.get_transformer_layers()
        text_cfg = self._get_text_config()
        snapshot: Dict[str, Any] = {
            "altup_active_idx": int(getattr(text_cfg, "altup_active_idx", 0)),
            "altup_correct_scale": bool(getattr(text_cfg, "altup_correct_scale", True)),
            "num_kv_shared_layers": int(getattr(text_cfg, "num_kv_shared_layers", 0)),
            "activation_sparsity_pattern": [
                float(getattr(layer.mlp, "activation_sparsity", 0.0)) for layer in layers
            ],
            "attn_flags": [
                (
                    bool(getattr(layer.self_attn, "is_kv_shared_layer", False)),
                    getattr(layer.self_attn, "kv_shared_layer_index", None),
                    bool(getattr(layer.self_attn, "store_full_length_kv", False)),
                )
                for layer in layers
            ],
        }

        try:
            self._set_text_runtime_overrides(overrides)
            yield
        finally:
            restore_overrides = {
                "altup_active_idx": snapshot["altup_active_idx"],
                "altup_correct_scale": snapshot["altup_correct_scale"],
                "num_kv_shared_layers": snapshot["num_kv_shared_layers"],
                "activation_sparsity_pattern": snapshot["activation_sparsity_pattern"],
            }
            self._set_text_runtime_overrides(restore_overrides)

    def _set_text_runtime_overrides(self, overrides: dict[str, Any]) -> None:
        if not overrides:
            return
        layers = self.get_transformer_layers()
        text_cfg = self._get_text_config()

        if "altup_active_idx" in overrides:
            text_cfg.altup_active_idx = int(overrides["altup_active_idx"])
        if "altup_correct_scale" in overrides:
            text_cfg.altup_correct_scale = bool(overrides["altup_correct_scale"])
        if "activation_sparsity_pattern" in overrides:
            asp = overrides["activation_sparsity_pattern"]
            if isinstance(asp, (int, float)):
                asp_values = [float(asp)] * len(layers)
            else:
                asp_values = [float(x) for x in asp]
                if len(asp_values) != len(layers):
                    raise ValueError(
                        "activation_sparsity_pattern must match number of layers: "
                        f"expected {len(layers)}, got {len(asp_values)}"
                    )
            text_cfg.activation_sparsity_pattern = asp_values
            for i, layer in enumerate(layers):
                layer.mlp.activation_sparsity = asp_values[i]

        if "num_kv_shared_layers" in overrides:
            self._reconfigure_kv_sharing(int(overrides["num_kv_shared_layers"]))

    def _reconfigure_kv_sharing(self, num_kv_shared_layers: int) -> None:
        layers = self.get_transformer_layers()
        text_cfg = self._get_text_config()
        num_layers = len(layers)
        if num_kv_shared_layers < 0 or num_kv_shared_layers >= num_layers:
            raise ValueError(
                f"num_kv_shared_layers must be in [0, {num_layers - 1}], got {num_kv_shared_layers}"
            )

        text_cfg.num_kv_shared_layers = num_kv_shared_layers
        first_kv_shared_layer_idx = num_layers - num_kv_shared_layers
        prev_layers = list(text_cfg.layer_types[:first_kv_shared_layer_idx])

        for layer_idx, layer in enumerate(layers):
            attn = layer.self_attn
            is_shared = layer_idx >= first_kv_shared_layer_idx > 0
            attn.is_kv_shared_layer = is_shared
            attn.layer_idx = layer_idx

            if is_shared:
                layer_type = text_cfg.layer_types[layer_idx]
                if layer_type not in prev_layers:
                    raise ValueError(
                        f"Cannot share KV for layer {layer_idx} ({layer_type}) because no non-shared peer exists"
                    )
                attn.kv_shared_layer_index = len(prev_layers) - 1 - prev_layers[::-1].index(layer_type)
                attn.store_full_length_kv = False
            else:
                attn.kv_shared_layer_index = None
                layer_type = text_cfg.layer_types[layer_idx]
                if layer_type in prev_layers:
                    last_idx = len(prev_layers) - 1 - prev_layers[::-1].index(layer_type)
                    attn.store_full_length_kv = layer_idx == last_idx
                else:
                    attn.store_full_length_kv = False

    def _get_text_config(self):
        if hasattr(self.model.config, "text_config"):
            return self.model.config.text_config
        return self.model.config

    @staticmethod
    def _resolve_parent_attr(root: Any, dotted: str) -> tuple[Any, str]:
        parts = dotted.split(".")
        obj = root
        for part in parts[:-1]:
            obj = getattr(obj, part)
        return obj, parts[-1]
