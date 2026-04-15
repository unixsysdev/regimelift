#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from helmas3n.scripts.train_uplift import _build_model, _load_yaml
from helmas3n.src.data.paired_dataset import PairedStateDataset
from helmas3n.src.eval.decode_resume import resume_with_kv_cache, resume_with_residual_patch
from helmas3n.src.eval.handoff_metrics import continuation_match_rate
from helmas3n.src.gemma.runner import GemmaRunner
from helmas3n.src.gemma.state_extract import load_extraction_config


@dataclass
class PromptCacheItem:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    low_states: torch.Tensor
    full_states: torch.Tensor
    full_tokens: torch.Tensor
    low_native_tokens: torch.Tensor
    low_to_full_tokens: torch.Tensor
    seed: int


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_prompts_from_extract_cfg(extract_cfg_path: Path, max_prompts: int) -> list[str]:
    cfg = yaml.safe_load(extract_cfg_path.read_text())
    prompts_path = Path(cfg["data"]["prompts_path"])
    if not prompts_path.is_absolute():
        prompts_path = (extract_cfg_path.parent / prompts_path).resolve()
    prompt_field = cfg["data"].get("prompt_field", "prompt")

    prompts = []
    if prompts_path.suffix == ".jsonl":
        for line in prompts_path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            prompts.append(str(item.get(prompt_field, item.get("prompt", ""))))
    elif prompts_path.suffix == ".json":
        raw = json.loads(prompts_path.read_text())
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    prompts.append(str(item.get(prompt_field, item.get("prompt", ""))))
                else:
                    prompts.append(str(item))
    else:
        for line in prompts_path.read_text().splitlines():
            if line.strip():
                prompts.append(line.strip())

    return [p for p in prompts if p][:max_prompts]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            cols.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def _parse_layers(num_layers: int, layer_stride: int, layer_list: str | None) -> list[int]:
    if layer_list:
        vals = []
        for tok in layer_list.split(","):
            tok = tok.strip()
            if not tok:
                continue
            v = int(tok)
            if v < 0:
                v = num_layers + v
            if v < 0 or v >= num_layers:
                raise ValueError(f"Layer index out of range: {tok} -> {v}, num_layers={num_layers}")
            vals.append(v)
        return sorted(set(vals))

    layers = list(range(0, num_layers, max(1, layer_stride)))
    if (num_layers - 1) not in layers:
        layers.append(num_layers - 1)
    return sorted(set(layers))


def _parse_oracle_groups(spec: str, num_layers: int, layers: list[int]) -> list[tuple[str, list[int]]]:
    if not spec.strip():
        return []
    groups: list[tuple[str, list[int]]] = []
    for raw in spec.split(";"):
        item = raw.strip()
        if not item:
            continue
        if ":" in item:
            name, rhs = item.split(":", 1)
            name = name.strip()
            rhs = rhs.strip()
        else:
            name, rhs = item, item

        if rhs == "last1":
            idxs = [num_layers - 1]
        elif rhs == "last2":
            idxs = [num_layers - 2, num_layers - 1]
        elif rhs == "last4":
            idxs = list(range(max(0, num_layers - 4), num_layers))
        elif rhs == "stride":
            idxs = list(layers)
        else:
            idxs = []
            for tok in rhs.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                v = int(tok)
                if v < 0:
                    v = num_layers + v
                if v < 0 or v >= num_layers:
                    raise ValueError(f"oracle group layer out of range: {tok} -> {v}, num_layers={num_layers}")
                idxs.append(v)
        groups.append((name, sorted(set(idxs))))
    return groups


def _load_model_for_method(
    method: str,
    train_cfg: dict[str, Any],
    paired_root: Path,
    checkpoint_root: Path,
) -> torch.nn.Module:
    if method == "identity":
        model = _build_model(
            kind="identity",
            num_layers=int(train_cfg["model"]["num_layers_hint"]),
            hidden_dim=int(train_cfg["model"]["hidden_dim_hint"]),
            model_cfg=train_cfg["model"],
        )
        model.eval()
        return model

    ckpt_path = checkpoint_root / method / "last.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for method={method}: {ckpt_path}")

    dataset = PairedStateDataset(str(paired_root), target="residual")
    sample = dataset[0]
    hidden_dim = int(sample["source_state"].numel())
    num_layers = max(int(dataset.info.num_layers), int(sample["layer"].item()) + 1)
    model = _build_model(kind=method, num_layers=num_layers, hidden_dim=hidden_dim, model_cfg=train_cfg["model"])
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def _build_prompt_cache(
    runner: GemmaRunner,
    prompts: list[str],
    max_length: int,
    max_h: int,
    base_seed: int,
) -> list[PromptCacheItem]:
    cache: list[PromptCacheItem] = []
    total = len(prompts)
    num_layers = runner.get_num_layers()

    for i, prompt in enumerate(prompts, start=1):
        if i == 1 or i % 10 == 0 or i == total:
            print(f"[layer-sweep] caching prompt {i}/{total}")
        prompt_seed = int(base_seed + i * 1009)
        _set_seed(prompt_seed)
        input_ids, attention_mask = runner.tokenize(prompt, max_length=max_length)
        low = runner.forward_prefix(input_ids, attention_mask, regime="low", capture_kv=True, capture_logits=True)
        _set_seed(prompt_seed + 1)
        full = runner.forward_prefix(input_ids, attention_mask, regime="full", capture_kv=True, capture_logits=True)

        low_states = torch.stack(
            [runner.normalize_hidden_state(low.hidden_states[layer + 1])[0, -1, :].detach().cpu() for layer in range(num_layers)],
            dim=0,
        )
        full_states = torch.stack(
            [runner.normalize_hidden_state(full.hidden_states[layer + 1])[0, -1, :].detach().cpu() for layer in range(num_layers)],
            dim=0,
        )

        full_tokens = resume_with_kv_cache(
            runner=runner,
            uplifted_past_key_values=full.past_key_values,
            last_token_ids=input_ids[:, -1:],
            regime="full",
            max_new_tokens=max_h,
            seed_token_ids=full.logits[:, -1, :].argmax(dim=-1, keepdim=True),
        ).cpu()
        low_native_tokens = resume_with_kv_cache(
            runner=runner,
            uplifted_past_key_values=low.past_key_values,
            last_token_ids=input_ids[:, -1:],
            regime="low",
            max_new_tokens=max_h,
            seed_token_ids=low.logits[:, -1, :].argmax(dim=-1, keepdim=True),
        ).cpu()
        low_to_full_tokens = resume_with_kv_cache(
            runner=runner,
            uplifted_past_key_values=low.past_key_values,
            last_token_ids=input_ids[:, -1:],
            regime="full",
            max_new_tokens=max_h,
            seed_token_ids=low.logits[:, -1, :].argmax(dim=-1, keepdim=True),
        ).cpu()

        cache.append(
            PromptCacheItem(
                input_ids=input_ids.cpu(),
                attention_mask=attention_mask.cpu(),
                low_states=low_states,
                full_states=full_states,
                full_tokens=full_tokens,
                low_native_tokens=low_native_tokens,
                low_to_full_tokens=low_to_full_tokens,
                seed=prompt_seed,
            )
        )

    return cache


def _aggregate_baseline(
    runner: GemmaRunner,
    cache: list[PromptCacheItem],
    horizons: list[int],
) -> dict[str, float]:
    out: dict[str, float] = {}
    n = max(len(cache), 1)
    max_h = max(horizons)
    for h in horizons:
        low_native = 0.0
        low_to_full_cached = 0.0
        low_to_full_live_nopatch = 0.0
        for item in cache:
            low_native += float(continuation_match_rate(item.full_tokens[:, :h], item.low_native_tokens[:, :h]).item())
            low_to_full_cached += float(continuation_match_rate(item.full_tokens[:, :h], item.low_to_full_tokens[:, :h]).item())
            _set_seed(item.seed)
            live_nopatch = resume_with_residual_patch(
                runner=runner,
                input_ids=item.input_ids.to(runner.device),
                attention_mask=item.attention_mask.to(runner.device),
                patch_map={},
                regime="low",
                decode_regime="full",
                max_new_tokens=max_h,
            ).cpu()
            low_to_full_live_nopatch += float(continuation_match_rate(item.full_tokens[:, :h], live_nopatch[:, :h]).item())
        out[f"low_native_h{h}"] = low_native / n
        out[f"low_to_full_cached_h{h}"] = low_to_full_cached / n
        out[f"low_to_full_live_nopatch_h{h}"] = low_to_full_live_nopatch / n
    return out


def _evaluate_single_layer(
    runner: GemmaRunner,
    cache: list[PromptCacheItem],
    method: str,
    layer: int,
    horizons: list[int],
    model: torch.nn.Module | None,
) -> dict[str, Any]:
    max_h = max(horizons)
    device = runner.device
    layer_t = torch.tensor([layer], device=device)
    total = len(cache)

    sums = {h: 0.0 for h in horizons}
    next_top1 = 0.0

    for i, item in enumerate(cache, start=1):
        if i == 1 or i % 10 == 0 or i == total:
            print(f"[layer-sweep] {method} layer={layer}: prompt {i}/{total}")
        if method == "oracle_full_state":
            patch_vec = item.full_states[layer]
        elif method == "identity":
            patch_vec = item.low_states[layer]
        else:
            assert model is not None
            with torch.no_grad():
                low_state = item.low_states[layer].to(device).float().unsqueeze(0)
                pred = model(low_state, layer_t)
                patch_vec = pred.squeeze(0).detach().cpu()

        _set_seed(item.seed)
        uplift = resume_with_residual_patch(
            runner=runner,
            input_ids=item.input_ids.to(device),
            attention_mask=item.attention_mask.to(device),
            patch_map={layer: patch_vec},
            regime="low",
            decode_regime="full",
            max_new_tokens=max_h,
        ).cpu()

        for h in horizons:
            sums[h] += float(continuation_match_rate(item.full_tokens[:, :h], uplift[:, :h]).item())
        if uplift.numel() > 0 and item.full_tokens.numel() > 0:
            next_top1 += float((uplift[:, 0] == item.full_tokens[:, 0]).float().mean().item())

    denom = max(total, 1)
    row: dict[str, Any] = {
        "method": method,
        "layer": layer,
        "prompt_count": total,
        "next_token_top1_uplift_vs_full_live": next_top1 / denom,
    }
    for h in horizons:
        row[f"cont_match_h{h}_uplift_vs_full"] = sums[h] / denom
    return row


def _evaluate_oracle_group(
    runner: GemmaRunner,
    cache: list[PromptCacheItem],
    group_name: str,
    layers: list[int],
    horizons: list[int],
) -> dict[str, Any]:
    max_h = max(horizons)
    device = runner.device
    total = len(cache)

    sums = {h: 0.0 for h in horizons}
    next_top1 = 0.0
    layer_str = ",".join(str(x) for x in layers)
    for i, item in enumerate(cache, start=1):
        if i == 1 or i % 10 == 0 or i == total:
            print(f"[layer-sweep] oracle_group={group_name} layers={layer_str}: prompt {i}/{total}")
        patch_map = {l: item.full_states[l] for l in layers}
        _set_seed(item.seed)
        uplift = resume_with_residual_patch(
            runner=runner,
            input_ids=item.input_ids.to(device),
            attention_mask=item.attention_mask.to(device),
            patch_map=patch_map,
            regime="low",
            decode_regime="full",
            max_new_tokens=max_h,
        ).cpu()

        for h in horizons:
            sums[h] += float(continuation_match_rate(item.full_tokens[:, :h], uplift[:, :h]).item())
        if uplift.numel() > 0 and item.full_tokens.numel() > 0:
            next_top1 += float((uplift[:, 0] == item.full_tokens[:, 0]).float().mean().item())

    denom = max(total, 1)
    row: dict[str, Any] = {
        "method": "oracle_group",
        "group_name": group_name,
        "layers": layer_str,
        "prompt_count": total,
        "next_token_top1_uplift_vs_full_live": next_top1 / denom,
    }
    for h in horizons:
        row[f"cont_match_h{h}_uplift_vs_full"] = sums[h] / denom
    return row


def _layer_residual_stats(cache: list[PromptCacheItem], layer: int) -> tuple[float, float]:
    cos_vals = []
    mse_vals = []
    for item in cache:
        low = item.low_states[layer]
        full = item.full_states[layer]
        cos_vals.append(float(F.cosine_similarity(low.unsqueeze(0), full.unsqueeze(0), dim=-1).item()))
        mse_vals.append(float(F.mse_loss(low, full).item()))
    denom = max(len(cos_vals), 1)
    return sum(cos_vals) / denom, sum(mse_vals) / denom


def main() -> None:
    parser = argparse.ArgumentParser(description="Live handoff layer sweep for HeLMAS-3n")
    parser.add_argument("--extract-config", type=str, default="configs/extract_killtest40.yaml")
    parser.add_argument("--train-template", type=str, default="configs/train_residual.yaml")
    parser.add_argument("--checkpoint-root", type=str, default="artifacts/reports/killtest_v3/checkpoints")
    parser.add_argument("--out-dir", type=str, default="artifacts/reports/layer_sweep_v1")
    parser.add_argument("--max-prompts", type=int, default=40)
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated layer ids. Defaults to stride from extract config.")
    parser.add_argument("--methods", type=str, default="identity,mlp,oracle_full_state")
    parser.add_argument("--horizons", type=str, default="1,4,8,16")
    parser.add_argument(
        "--oracle-groups",
        type=str,
        default="",
        help="Optional semicolon-separated oracle multi-layer groups, e.g. 'last1:last1;last2:last2;late4:31,32,33,34'",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    extract_cfg_path = Path(args.extract_config)
    if not extract_cfg_path.is_absolute():
        extract_cfg_path = (script_dir.parent / args.extract_config).resolve()
    train_template_path = Path(args.train_template)
    if not train_template_path.is_absolute():
        train_template_path = (script_dir.parent / args.train_template).resolve()
    checkpoint_root = Path(args.checkpoint_root)
    if not checkpoint_root.is_absolute():
        checkpoint_root = (script_dir.parent / args.checkpoint_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (script_dir.parent / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    extract_cfg = load_extraction_config(extract_cfg_path)
    paired_root = Path(extract_cfg.out_dir).resolve()

    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    if not horizons:
        raise ValueError("No horizons provided")
    horizons = sorted(set(horizons))

    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    if not methods:
        raise ValueError("No methods provided")

    base_train_cfg = _load_yaml(train_template_path)
    ds = PairedStateDataset(str(paired_root), target="residual")
    sample = ds[0]
    base_train_cfg["model"]["hidden_dim_hint"] = int(sample["source_state"].numel())
    base_train_cfg["model"]["num_layers_hint"] = int(max(ds.info.num_layers, int(sample["layer"].item()) + 1))

    runner = GemmaRunner(
        model_name=extract_cfg.model_name,
        device=extract_cfg.device,
        dtype=extract_cfg.dtype,
        trust_remote_code=extract_cfg.trust_remote_code,
        regimes=extract_cfg.regimes,
        model_load_overrides=extract_cfg.model_load_overrides,
        text_only=extract_cfg.text_only,
    )

    num_layers = runner.get_num_layers()
    layers = _parse_layers(num_layers=num_layers, layer_stride=extract_cfg.layer_stride, layer_list=args.layers)
    oracle_groups = _parse_oracle_groups(spec=args.oracle_groups, num_layers=num_layers, layers=layers)
    prompts = _load_prompts_from_extract_cfg(extract_cfg_path, max_prompts=args.max_prompts)

    print(
        json.dumps(
            {
                "prompts": len(prompts),
                "layers": layers,
                "methods": methods,
                "oracle_groups": [{"name": n, "layers": g} for n, g in oracle_groups],
                "horizons": horizons,
            },
            indent=2,
        )
    )

    cache = _build_prompt_cache(
        runner=runner,
        prompts=prompts,
        max_length=extract_cfg.max_length,
        max_h=max(horizons),
        base_seed=extract_cfg.seed,
    )
    baseline = _aggregate_baseline(runner=runner, cache=cache, horizons=horizons)

    models: dict[str, torch.nn.Module] = {}
    for method in methods:
        if method in {"oracle_full_state", "identity"}:
            continue
        model = _load_model_for_method(
            method=method,
            train_cfg=base_train_cfg,
            paired_root=paired_root,
            checkpoint_root=checkpoint_root,
        )
        model.to(runner.device)
        models[method] = model

    rows: list[dict[str, Any]] = []
    for method in methods:
        if method == "oracle_full_state":
            model = None
        elif method == "identity":
            model = None
        else:
            model = models[method]

        for layer in layers:
            row = _evaluate_single_layer(
                runner=runner,
                cache=cache,
                method=method,
                layer=layer,
                horizons=horizons,
                model=model,
            )
            low_full_cos, low_full_mse = _layer_residual_stats(cache, layer)
            row["residual_cosine_low_vs_full_layer"] = low_full_cos
            row["residual_mse_low_vs_full_layer"] = low_full_mse
            for h in horizons:
                row[f"cont_match_h{h}_low_to_full_no_patch"] = baseline[f"low_to_full_live_nopatch_h{h}"]
                row[f"cont_match_h{h}_low_to_full_cached"] = baseline[f"low_to_full_cached_h{h}"]
                row[f"cont_match_h{h}_low_native"] = baseline[f"low_native_h{h}"]
                row[f"delta_h{h}"] = (
                    row[f"cont_match_h{h}_uplift_vs_full"] - baseline[f"low_to_full_live_nopatch_h{h}"]
                )
            rows.append(row)

    for group_name, group_layers in oracle_groups:
        row = _evaluate_oracle_group(
            runner=runner,
            cache=cache,
            group_name=group_name,
            layers=group_layers,
            horizons=horizons,
        )
        for h in horizons:
            row[f"cont_match_h{h}_low_to_full_no_patch"] = baseline[f"low_to_full_live_nopatch_h{h}"]
            row[f"cont_match_h{h}_low_to_full_cached"] = baseline[f"low_to_full_cached_h{h}"]
            row[f"cont_match_h{h}_low_native"] = baseline[f"low_native_h{h}"]
            row[f"delta_h{h}"] = row[f"cont_match_h{h}_uplift_vs_full"] - baseline[f"low_to_full_live_nopatch_h{h}"]
        rows.append(row)

    curves = []
    for row in rows:
        layer_tag = row.get("layer")
        if row.get("method") == "oracle_group":
            layer_tag = row.get("layers")
        for h in horizons:
            curves.append(
                {
                    "method": row["method"],
                    "layer": layer_tag,
                    "horizon": h,
                    "match_uplift_vs_full": row.get(f"cont_match_h{h}_uplift_vs_full"),
                    "match_low_to_full_no_patch": row.get(f"cont_match_h{h}_low_to_full_no_patch"),
                    "delta_over_low_to_full_no_patch": row.get(f"delta_h{h}"),
                }
            )

    out = {
        "out_dir": str(out_dir),
        "num_rows": len(rows),
        "num_prompts": len(prompts),
        "layers": layers,
        "methods": methods,
        "horizons": horizons,
        "baseline": baseline,
    }
    (out_dir / "layer_sweep_rows.json").write_text(json.dumps(rows, indent=2))
    _write_csv(out_dir / "layer_sweep_rows.csv", rows)
    _write_csv(out_dir / "layer_sweep_curves.csv", curves)
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
