#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from helmas3n.src.data.paired_dataset import PairedStateDataset
from helmas3n.src.eval.decode_resume import resume_with_kv_cache, resume_with_residual_patch
from helmas3n.src.eval.handoff_metrics import continuation_match_rate
from helmas3n.src.gemma.runner import GemmaRunner
from helmas3n.src.gemma.state_extract import load_extraction_config
from helmas3n.scripts.train_uplift import _build_model, _load_yaml


SITE_SPECS: dict[str, list[int]] = {
    "layer34": [34],
    "layer16": [16],
    "stride": [0, 4, 8, 12, 16, 20, 24, 28, 32, 34],
}
SPANS = [1, 4, 8]
METHODS = ["low_to_full_no_patch", "identity", "mlp", "oracle"]
HORIZONS = [1, 4, 8, 16]


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

    prompts: list[str] = []
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
            if key not in seen:
                seen.add(key)
                cols.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def _load_mlp_model(train_cfg: dict[str, Any], paired_root: Path, checkpoint_root: Path) -> torch.nn.Module:
    dataset = PairedStateDataset(str(paired_root), target="residual")
    sample = dataset[0]
    hidden_dim = int(sample["source_state"].numel())
    num_layers = max(int(dataset.info.num_layers), int(sample["layer"].item()) + 1)
    model = _build_model(
        kind="mlp",
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        model_cfg=train_cfg["model"],
    )
    ckpt_path = checkpoint_root / "mlp" / "last.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing MLP checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def _expand_patch_tensor(base_state: torch.Tensor, replacement: torch.Tensor, span: int) -> torch.Tensor:
    patch = base_state.clone()
    eff_span = min(int(span), int(patch.size(0)))
    if eff_span > 0:
        patch[-eff_span:] = replacement[-eff_span:]
    return patch


def _predict_suffix_patches(
    model: torch.nn.Module,
    low_states_by_layer: dict[int, torch.Tensor],
    site_layers: list[int],
    span: int,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    inputs: list[torch.Tensor] = []
    layer_ids: list[int] = []
    layer_sizes: dict[int, int] = {}

    for layer in site_layers:
        low_state = low_states_by_layer[layer]
        eff_span = min(int(span), int(low_state.size(0)))
        layer_sizes[layer] = eff_span
        if eff_span <= 0:
            continue
        inputs.append(low_state[-eff_span:])
        layer_ids.extend([layer] * eff_span)

    if not inputs:
        return {layer: low_states_by_layer[layer].clone() for layer in site_layers}

    flat_inputs = torch.cat(inputs, dim=0).to(device=device, dtype=torch.float32)
    flat_layers = torch.tensor(layer_ids, device=device, dtype=torch.long)
    with torch.no_grad():
        pred = model(flat_inputs, flat_layers).detach().cpu()

    out: dict[int, torch.Tensor] = {}
    offset = 0
    for layer in site_layers:
        eff_span = layer_sizes[layer]
        base = low_states_by_layer[layer].clone()
        if eff_span > 0:
            base[-eff_span:] = pred[offset : offset + eff_span]
        out[layer] = base
        offset += eff_span
    return out


def _match_metrics(full_tokens: torch.Tensor, candidate_tokens: torch.Tensor, horizons: list[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for h in horizons:
        out[f"match_h{h}"] = float(continuation_match_rate(full_tokens[:, :h], candidate_tokens[:, :h]).item())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Suffix-span sweep for HeLMAS-3n live handoff")
    parser.add_argument("--extract-config", type=str, default="configs/extract_killtest40.yaml")
    parser.add_argument("--train-template", type=str, default="configs/train_residual.yaml")
    parser.add_argument("--checkpoint-root", type=str, default="artifacts/reports/killtest_v3/checkpoints")
    parser.add_argument("--out-dir", type=str, default="artifacts/reports/suffix_span_sweep_v1")
    parser.add_argument("--max-prompts", type=int, default=40)
    parser.add_argument("--prompt-limit", type=int, default=None)
    parser.add_argument("--horizons", type=str, default="1,4,8,16")
    parser.add_argument("--spans", type=str, default="1,4,8")
    parser.add_argument("--sites", type=str, default="layer34,layer16,stride")
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
    train_cfg = _load_yaml(train_template_path)

    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    spans = [int(x.strip()) for x in args.spans.split(",") if x.strip()]
    sites = [x.strip() for x in args.sites.split(",") if x.strip()]
    if not horizons or not spans or not sites:
        raise ValueError("horizons, spans, and sites must be non-empty")
    horizons = sorted(set(horizons))
    spans = sorted(set(spans))

    prompt_count = args.prompt_limit if args.prompt_limit is not None else args.max_prompts
    prompts = _load_prompts_from_extract_cfg(extract_cfg_path, max_prompts=prompt_count)

    runner = GemmaRunner(
        model_name=extract_cfg.model_name,
        device=extract_cfg.device,
        dtype=extract_cfg.dtype,
        trust_remote_code=extract_cfg.trust_remote_code,
        regimes=extract_cfg.regimes,
        model_load_overrides=extract_cfg.model_load_overrides,
        text_only=extract_cfg.text_only,
    )

    mlp_model = _load_mlp_model(train_cfg=train_cfg, paired_root=paired_root, checkpoint_root=checkpoint_root)
    mlp_model.to(runner.device)

    print(
        json.dumps(
            {
                "prompts": len(prompts),
                "sites": sites,
                "site_layers": {site: SITE_SPECS[site] for site in sites},
                "spans": spans,
                "horizons": horizons,
            },
            indent=2,
        )
    )

    prompt_rows: list[dict[str, Any]] = []
    summary_sums: dict[tuple[str, int, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    summary_counts: dict[tuple[str, int, str], int] = defaultdict(int)

    max_h = max(horizons)
    num_layers = runner.get_num_layers()
    base_seed = extract_cfg.seed

    for prompt_idx, prompt in enumerate(prompts, start=1):
        if prompt_idx == 1 or prompt_idx % 10 == 0 or prompt_idx == len(prompts):
            print(f"[suffix-sweep] prompt {prompt_idx}/{len(prompts)}")
        prompt_seed = int(base_seed + prompt_idx * 1009)

        _set_seed(prompt_seed)
        input_ids, attention_mask = runner.tokenize(prompt, max_length=extract_cfg.max_length)

        _set_seed(prompt_seed)
        low = runner.forward_prefix(input_ids, attention_mask, regime="low", capture_kv=True, capture_logits=True)
        _set_seed(prompt_seed)
        full = runner.forward_prefix(input_ids, attention_mask, regime="full", capture_kv=True, capture_logits=True)

        full_tokens = resume_with_kv_cache(
            runner=runner,
            uplifted_past_key_values=full.past_key_values,
            last_token_ids=input_ids[:, -1:],
            regime="full",
            max_new_tokens=max_h,
            seed_token_ids=full.logits[:, -1, :].argmax(dim=-1, keepdim=True),
        ).cpu()

        _set_seed(prompt_seed)
        no_patch_tokens = resume_with_residual_patch(
            runner=runner,
            input_ids=input_ids,
            attention_mask=attention_mask,
            patch_map={},
            regime="low",
            decode_regime="full",
            max_new_tokens=max_h,
        ).cpu()

        baseline_metrics = _match_metrics(full_tokens, no_patch_tokens, horizons)

        low_states_by_layer: dict[int, torch.Tensor] = {}
        full_states_by_layer: dict[int, torch.Tensor] = {}
        for layer in sorted({layer for site in sites for layer in SITE_SPECS[site]}):
            if layer < 0 or layer >= num_layers:
                raise ValueError(f"Layer {layer} out of range for model with {num_layers} layers")
            low_states_by_layer[layer] = runner.normalize_hidden_state(low.hidden_states[layer + 1])[0].detach().cpu()
            full_states_by_layer[layer] = runner.normalize_hidden_state(full.hidden_states[layer + 1])[0].detach().cpu()

        for site in sites:
            site_layers = SITE_SPECS[site]
            for span in spans:
                oracle_patch_map: dict[int, torch.Tensor] = {}
                for layer in site_layers:
                    oracle_patch_map[layer] = _expand_patch_tensor(
                        base_state=low_states_by_layer[layer],
                        replacement=full_states_by_layer[layer],
                        span=span,
                    )

                _set_seed(prompt_seed)
                oracle_tokens = resume_with_residual_patch(
                    runner=runner,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    patch_map=oracle_patch_map,
                    regime="low",
                    decode_regime="full",
                    max_new_tokens=max_h,
                ).cpu()
                oracle_metrics = _match_metrics(full_tokens, oracle_tokens, horizons)

                # Control row. This is the live no-patch baseline, duplicated per site/span for readability.
                ctrl_row = {
                    "prompt_index": prompt_idx,
                    "site": site,
                    "span": span,
                    "method": "low_to_full_no_patch",
                    "prompt_count": len(prompts),
                }
                for h in horizons:
                    ctrl_row[f"match_h{h}"] = baseline_metrics[f"match_h{h}"]
                    ctrl_row[f"no_patch_h{h}"] = baseline_metrics[f"match_h{h}"]
                    ctrl_row[f"oracle_h{h}"] = oracle_metrics[f"match_h{h}"]
                    ctrl_row[f"delta_h{h}"] = 0.0
                    ctrl_row[f"closure_h{h}"] = 0.0
                prompt_rows.append(ctrl_row)
                _accumulate(summary_sums, summary_counts, ctrl_row, horizons)

                # Identity row is the no-patch control, duplicated so the requested table has the control row.
                id_row = copy.deepcopy(ctrl_row)
                id_row["method"] = "identity"
                prompt_rows.append(id_row)
                _accumulate(summary_sums, summary_counts, id_row, horizons)

                # Oracle row.
                oracle_row = {
                    "prompt_index": prompt_idx,
                    "site": site,
                    "span": span,
                    "method": "oracle",
                    "prompt_count": len(prompts),
                }
                for h in horizons:
                    oracle_row[f"match_h{h}"] = oracle_metrics[f"match_h{h}"]
                    oracle_row[f"no_patch_h{h}"] = baseline_metrics[f"match_h{h}"]
                    oracle_row[f"oracle_h{h}"] = oracle_metrics[f"match_h{h}"]
                    oracle_row[f"delta_h{h}"] = oracle_metrics[f"match_h{h}"] - baseline_metrics[f"match_h{h}"]
                    oracle_row[f"closure_h{h}"] = _closure(
                        method=oracle_metrics[f"match_h{h}"],
                        no_patch=baseline_metrics[f"match_h{h}"],
                        oracle=oracle_metrics[f"match_h{h}"],
                    )
                prompt_rows.append(oracle_row)
                _accumulate(summary_sums, summary_counts, oracle_row, horizons)

                # MLP row.
                mlp_patch_map: dict[int, torch.Tensor] = {}
                for layer in site_layers:
                    mlp_patch_map[layer] = low_states_by_layer[layer]
                # Predict the suffix for each selected layer.
                suffix_predictions = _predict_suffix_patches(
                    model=mlp_model,
                    low_states_by_layer=low_states_by_layer,
                    site_layers=site_layers,
                    span=span,
                    device=runner.device,
                )
                mlp_patch_map.update(suffix_predictions)

                _set_seed(prompt_seed)
                mlp_tokens = resume_with_residual_patch(
                    runner=runner,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    patch_map=mlp_patch_map,
                    regime="low",
                    decode_regime="full",
                    max_new_tokens=max_h,
                ).cpu()
                mlp_metrics = _match_metrics(full_tokens, mlp_tokens, horizons)

                mlp_row = {
                    "prompt_index": prompt_idx,
                    "site": site,
                    "span": span,
                    "method": "mlp",
                    "prompt_count": len(prompts),
                }
                for h in horizons:
                    mlp_row[f"match_h{h}"] = mlp_metrics[f"match_h{h}"]
                    mlp_row[f"no_patch_h{h}"] = baseline_metrics[f"match_h{h}"]
                    mlp_row[f"oracle_h{h}"] = oracle_metrics[f"match_h{h}"]
                    mlp_row[f"delta_h{h}"] = mlp_metrics[f"match_h{h}"] - baseline_metrics[f"match_h{h}"]
                    mlp_row[f"closure_h{h}"] = _closure(
                        method=mlp_metrics[f"match_h{h}"],
                        no_patch=baseline_metrics[f"match_h{h}"],
                        oracle=oracle_metrics[f"match_h{h}"],
                    )
                prompt_rows.append(mlp_row)
                _accumulate(summary_sums, summary_counts, mlp_row, horizons)

    summary_rows = _finalize_summary(summary_sums, summary_counts, horizons)
    summary_rows = sorted(summary_rows, key=lambda r: (r["site"], r["span"], METHODS.index(r["method"])))

    _write_csv(out_dir / "suffix_span_prompt_rows.csv", prompt_rows)
    _write_csv(out_dir / "suffix_span_summary.csv", summary_rows)
    _write_report(out_dir / "suffix_span_report.md", prompt_rows, summary_rows)
    (out_dir / "suffix_span_summary.json").write_text(
        json.dumps(
            {
                "extract_config": str(extract_cfg_path),
                "train_template": str(train_template_path),
                "checkpoint_root": str(checkpoint_root),
                "out_dir": str(out_dir),
                "num_prompts": len(prompts),
                "sites": sites,
                "site_layers": {site: SITE_SPECS[site] for site in sites},
                "spans": spans,
                "horizons": horizons,
                "rows": summary_rows,
            },
            indent=2,
        )
    )
    print(json.dumps({"out_dir": str(out_dir), "prompt_rows": len(prompt_rows), "summary_rows": len(summary_rows)}, indent=2))


def _closure(method: float, no_patch: float, oracle: float) -> float:
    denom = oracle - no_patch
    if abs(denom) < 1e-9:
        return 0.0
    return (method - no_patch) / denom


def _accumulate(
    summary_sums: dict[tuple[str, int, str], dict[str, float]],
    summary_counts: dict[tuple[str, int, str], int],
    row: dict[str, Any],
    horizons: list[int],
) -> None:
    key = (row["site"], int(row["span"]), row["method"])
    summary_counts[key] += 1
    sums = summary_sums[key]
    for h in horizons:
        for field in ["match", "no_patch", "oracle", "delta", "closure"]:
            value = row.get(f"{field}_h{h}")
            if value is None:
                continue
            sums[f"{field}_h{h}"] += float(value)


def _finalize_summary(
    summary_sums: dict[tuple[str, int, str], dict[str, float]],
    summary_counts: dict[tuple[str, int, str], int],
    horizons: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, sums in summary_sums.items():
        site, span, method = key
        count = max(summary_counts[key], 1)
        row: dict[str, Any] = {
            "site": site,
            "span": span,
            "method": method,
            "prompt_count": count,
        }
        for h in horizons:
            match = sums.get(f"match_h{h}", 0.0) / count
            no_patch = sums.get(f"no_patch_h{h}", 0.0) / count
            oracle = sums.get(f"oracle_h{h}", 0.0) / count
            delta = sums.get(f"delta_h{h}", 0.0) / count
            closure = sums.get(f"closure_h{h}", 0.0) / count
            row[f"mean_match_h{h}"] = match
            row[f"mean_no_patch_h{h}"] = no_patch
            row[f"mean_oracle_h{h}"] = oracle
            row[f"mean_delta_h{h}"] = delta
            row[f"mean_closure_h{h}"] = closure
            row[f"mean_headroom_h{h}"] = oracle - no_patch
        rows.append(row)
    return rows


def _write_report(path: Path, prompt_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    control_ok = True
    by_key: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in prompt_rows:
        key = (str(row["prompt_index"]), str(row["site"]), str(row["span"]))
        by_key.setdefault(key, {})[str(row["method"])] = row
    for rows in by_key.values():
        a = rows.get("low_to_full_no_patch")
        b = rows.get("identity")
        if a is None or b is None:
            control_ok = False
            break
        for h in HORIZONS:
            if a.get(f"match_h{h}") != b.get(f"match_h{h}"):
                control_ok = False
                break
        if not control_ok:
            break

    lines: list[str] = []
    lines.append("# Suffix Span Sweep Report")
    lines.append("")
    lines.append(f"- Prompt rows: {len(prompt_rows)}")
    lines.append(f"- Summary rows: {len(summary_rows)}")
    lines.append(f"- Identity control passed: {str(control_ok).lower()}")
    lines.append("")
    lines.append("## Best Oracle Rows")
    lines.append("| Horizon | Site | Span | Mean match | Mean no-patch | Mean delta |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for h in HORIZONS:
        sub = [r for r in summary_rows if r["method"] == "oracle"]
        best = max(sub, key=lambda r: float(r[f"mean_delta_h{h}"]))
        lines.append(
            f"| h{h} | {best['site']} | last{best['span']} | {float(best[f'mean_match_h{h}']):.6f} | "
            f"{float(best[f'mean_no_patch_h{h}']):.6f} | {float(best[f'mean_delta_h{h}']):.6f} |"
        )
    lines.append("")
    lines.append("## Best MLP Rows")
    lines.append("| Horizon | Site | Span | Mean match | Mean no-patch | Mean delta | Mean closure |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for h in HORIZONS:
        sub = [r for r in summary_rows if r["method"] == "mlp"]
        best = max(sub, key=lambda r: float(r[f"ratio_closure_h{h}"]))
        lines.append(
            f"| h{h} | {best['site']} | last{best['span']} | {float(best[f'mean_match_h{h}']):.6f} | "
            f"{float(best[f'mean_no_patch_h{h}']):.6f} | {float(best[f'mean_delta_h{h}']):.6f} | "
            f"{float(best[f'ratio_closure_h{h}']):.3f} |"
        )
    lines.append("")
    lines.append("## Span Effect")
    for site in SITE_SPECS:
        site_rows = [r for r in summary_rows if r["method"] == "oracle" and r["site"] == site]
        h8_best = max(site_rows, key=lambda r: float(r["mean_delta_h8"]))
        h16_best = max(site_rows, key=lambda r: float(r["mean_delta_h16"]))
        lines.append(
            f"- {site}: best oracle h8 is last{h8_best['span']} (delta {float(h8_best['mean_delta_h8']):.6f}); "
            f"best oracle h16 is last{h16_best['span']} (delta {float(h16_best['mean_delta_h16']):.6f})."
        )
    lines.append("")
    lines.append("## Files")
    lines.append("- [Prompt rows](./suffix_span_prompt_rows.csv)")
    lines.append("- [Summary rows](./suffix_span_summary.csv)")
    lines.append("- [Summary JSON](./suffix_span_summary.json)")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
