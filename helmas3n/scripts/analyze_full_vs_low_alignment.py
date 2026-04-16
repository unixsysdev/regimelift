#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from helmas3n.scripts.train_uplift import _build_model, _load_yaml
from helmas3n.src.gemma.hooks import residual_patch_context
from helmas3n.src.gemma.runner import GemmaRunner
from helmas3n.src.gemma.state_extract import load_extraction_config


def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = (PROJECT_ROOT / "helmas3n" / p).resolve()
    return p


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
    else:
        raw = json.loads(prompts_path.read_text())
        for item in raw:
            if isinstance(item, dict):
                prompts.append(str(item.get(prompt_field, item.get("prompt", ""))))
            else:
                prompts.append(str(item))
    return [p for p in prompts if p][:max_prompts]


def _run_low_prefill_with_patch(
    runner: GemmaRunner,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    patch_map: dict[int, torch.Tensor],
) -> Any:
    layers = runner.get_transformer_layers()
    regime_cfg = runner.regimes.get("low", {})
    runtime_overrides = regime_cfg.get("runtime_overrides", {})
    with (
        torch.no_grad(),
        runner._apply_model_overrides(regime_cfg.get("model_overrides", {})),
        runner._apply_text_runtime_overrides(regime_cfg.get("text_overrides", {})),
    ):
        with residual_patch_context(layers, patch_map):
            return runner.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
                **runtime_overrides,
            )


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _load_uplift_model(train_cfg_path: Path, checkpoint_path: Path, runner: GemmaRunner) -> torch.nn.Module:
    cfg = _load_yaml(train_cfg_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    dataset_info = ckpt.get("dataset_info", {})
    hidden_dim = int(dataset_info.get("hidden_dim", runner.get_hidden_dim()))
    num_layers = int(dataset_info.get("num_layers", runner.get_num_layers()))
    model = _build_model(
        kind=cfg["model"]["kind"],
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        model_cfg=cfg["model"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(runner.device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure how full-like vs low-like each handoff method is.")
    parser.add_argument("--extract-config", type=str, default="configs/extract_killtest40_holdout80.yaml")
    parser.add_argument("--layer", type=int, default=34)
    parser.add_argument("--max-prompts", type=int, default=80)
    parser.add_argument("--broad-train-config", type=str, default="configs/train_residual.yaml")
    parser.add_argument("--broad-checkpoint", type=str, required=True)
    parser.add_argument("--targeted-train-config", type=str, default="configs/train_residual_layer34_last1_short_horizon.yaml")
    parser.add_argument("--targeted-checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="artifacts/reports/full_vs_low_alignment_layer34_last1")
    args = parser.parse_args()

    extract_cfg_path = _resolve(args.extract_config)
    broad_train_cfg = _resolve(args.broad_train_config)
    broad_ckpt = _resolve(args.broad_checkpoint)
    targeted_train_cfg = _resolve(args.targeted_train_config)
    targeted_ckpt = _resolve(args.targeted_checkpoint)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extract_cfg = load_extraction_config(extract_cfg_path)
    prompts = _load_prompts_from_extract_cfg(extract_cfg_path, max_prompts=args.max_prompts)
    runner = GemmaRunner(
        model_name=extract_cfg.model_name,
        device=extract_cfg.device,
        dtype=extract_cfg.dtype,
        trust_remote_code=extract_cfg.trust_remote_code,
        regimes=extract_cfg.regimes,
        model_load_overrides=extract_cfg.model_load_overrides,
        text_only=extract_cfg.text_only,
    )

    broad_model = _load_uplift_model(broad_train_cfg, broad_ckpt, runner)
    targeted_model = _load_uplift_model(targeted_train_cfg, targeted_ckpt, runner)
    layer = int(args.layer)

    methods = ["no_patch", "reference_layer34", "broad_mlp", "targeted_layer34"]
    per_method: dict[str, dict[str, list[float]]] = {
        m: {
            "kl_full_to_method": [],
            "kl_low_to_method": [],
            "top1_match_to_full": [],
            "top1_match_to_low": [],
        }
        for m in methods
    }
    rows: list[dict[str, Any]] = []

    for i, prompt in enumerate(prompts, start=1):
        if i == 1 or i % 10 == 0 or i == len(prompts):
            print(f"[full-vs-low] prompt {i}/{len(prompts)}", flush=True)

        input_ids, attention_mask = runner.tokenize(prompt, max_length=extract_cfg.max_length)
        low = runner.forward_prefix(input_ids, attention_mask, regime="low", capture_kv=False, capture_logits=True)
        full = runner.forward_prefix(input_ids, attention_mask, regime="full", capture_kv=False, capture_logits=True)

        low_state = runner.normalize_hidden_state(low.hidden_states[layer + 1])[0, -1, :].detach().to(runner.device).float().unsqueeze(0)
        full_site_state = runner.normalize_hidden_state(full.hidden_states[layer + 1])[0, -1, :].detach().cpu()

        with torch.no_grad():
            broad_state = broad_model(low_state, torch.tensor([layer], device=runner.device)).squeeze(0).detach().cpu()
            targeted_state = targeted_model(low_state, torch.tensor([layer], device=runner.device)).squeeze(0).detach().cpu()

        candidates = {
            "no_patch": {},
            "reference_layer34": {layer: full_site_state},
            "broad_mlp": {layer: broad_state},
            "targeted_layer34": {layer: targeted_state},
        }

        full_logits = full.logits[:, -1, :].to(runner.device).float()
        low_logits = low.logits[:, -1, :].to(runner.device).float()
        full_probs = F.softmax(full_logits, dim=-1)
        low_probs = F.softmax(low_logits, dim=-1)
        full_top = full_logits.argmax(dim=-1)
        low_top = low_logits.argmax(dim=-1)

        for method, patch_map in candidates.items():
            out = _run_low_prefill_with_patch(
                runner=runner,
                input_ids=input_ids,
                attention_mask=attention_mask,
                patch_map=patch_map,
            )
            method_logits = out.logits[:, -1, :].to(runner.device).float()
            method_top = method_logits.argmax(dim=-1)
            kl_full = float(F.kl_div(F.log_softmax(method_logits, dim=-1), full_probs, reduction="batchmean").item())
            kl_low = float(F.kl_div(F.log_softmax(method_logits, dim=-1), low_probs, reduction="batchmean").item())
            top1_full = float((method_top == full_top).float().mean().item())
            top1_low = float((method_top == low_top).float().mean().item())

            rows.append(
                {
                    "prompt_idx": i,
                    "method": method,
                    "kl_full_to_method": kl_full,
                    "kl_low_to_method": kl_low,
                    "top1_match_to_full": top1_full,
                    "top1_match_to_low": top1_low,
                }
            )
            per_method[method]["kl_full_to_method"].append(kl_full)
            per_method[method]["kl_low_to_method"].append(kl_low)
            per_method[method]["top1_match_to_full"].append(top1_full)
            per_method[method]["top1_match_to_low"].append(top1_low)

    with (out_dir / "full_vs_low_rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary_rows: list[dict[str, Any]] = []
    for method in methods:
        summary_rows.append(
            {
                "method": method,
                "mean_top1_match_to_full": _mean(per_method[method]["top1_match_to_full"]),
                "mean_top1_match_to_low": _mean(per_method[method]["top1_match_to_low"]),
                "mean_kl_full_to_method": _mean(per_method[method]["kl_full_to_method"]),
                "mean_kl_low_to_method": _mean(per_method[method]["kl_low_to_method"]),
            }
        )

    summary = {
        "extract_config": str(extract_cfg_path),
        "num_prompts": len(prompts),
        "layer": layer,
        "rows": summary_rows,
    }
    (out_dir / "full_vs_low_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = ["# Full-like vs Low-like Diagnostic", ""]
    report.append(f"- prompts: {len(prompts)}")
    report.append(f"- layer: {layer}")
    report.append("")
    report.append("| method | top1 match to full | top1 match to low | KL(full||method) | KL(low||method) |")
    report.append("|---|---:|---:|---:|---:|")
    for row in summary_rows:
        report.append(
            f"| {row['method']} | {row['mean_top1_match_to_full']:.6f} | "
            f"{row['mean_top1_match_to_low']:.6f} | {row['mean_kl_full_to_method']:.6f} | "
            f"{row['mean_kl_low_to_method']:.6f} |"
        )
    (out_dir / "full_vs_low_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "summary_rows": summary_rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
