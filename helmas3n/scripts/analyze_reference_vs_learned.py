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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare learned vs reference patch behavior at layer34,last1.")
    parser.add_argument("--train-config", type=str, default="configs/train_residual_layer34_last1.yaml")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--extract-config", type=str, default="configs/extract_killtest40_holdout80.yaml")
    parser.add_argument("--layer", type=int, default=34)
    parser.add_argument("--max-prompts", type=int, default=80)
    parser.add_argument("--out-dir", type=str, default="artifacts/reports/reference_vs_learned_layer34_last1")
    args = parser.parse_args()

    train_cfg_path = _resolve(args.train_config)
    train_cfg = _load_yaml(train_cfg_path)
    checkpoint = _resolve(args.checkpoint) if args.checkpoint else Path(train_cfg["experiment"]["out_dir"]) / "last.pt"
    extract_cfg_path = _resolve(args.extract_config)
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

    ckpt = torch.load(checkpoint, map_location="cpu")
    dataset_info = ckpt.get("dataset_info", {})
    hidden_dim = int(dataset_info.get("hidden_dim", runner.get_hidden_dim()))
    num_layers = int(dataset_info.get("num_layers", runner.get_num_layers()))
    model = _build_model(
        kind=train_cfg["model"]["kind"],
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        model_cfg=train_cfg["model"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(runner.device)
    model.eval()

    methods = ["no_patch", "reference_patch", "learned_patch"]
    rows: list[dict[str, Any]] = []
    per_method: dict[str, dict[str, list[float]]] = {
        m: {"kl_full_to_method": [], "top1_match_to_full": [], "final_hidden_cos_to_full": []}
        for m in methods
    }
    layer = int(args.layer)

    for i, prompt in enumerate(prompts, start=1):
        if i == 1 or i % 10 == 0 or i == len(prompts):
            print(f"[diagnostic] prompt {i}/{len(prompts)}", flush=True)

        input_ids, attention_mask = runner.tokenize(prompt, max_length=extract_cfg.max_length)
        low = runner.forward_prefix(input_ids, attention_mask, regime="low", capture_kv=False, capture_logits=True)
        full = runner.forward_prefix(input_ids, attention_mask, regime="full", capture_kv=False, capture_logits=True)

        low_state = runner.normalize_hidden_state(low.hidden_states[layer + 1])[0, -1, :].detach().to(runner.device).float().unsqueeze(0)
        full_site_state = runner.normalize_hidden_state(full.hidden_states[layer + 1])[0, -1, :].detach().cpu()
        with torch.no_grad():
            learned_state = model(low_state, torch.tensor([layer], device=runner.device)).squeeze(0).detach().cpu()

        candidates = {
            "no_patch": {},
            "reference_patch": {layer: full_site_state},
            "learned_patch": {layer: learned_state},
        }

        full_logits = full.logits[:, -1, :].to(runner.device).float()
        full_probs = F.softmax(full_logits, dim=-1)
        full_hidden_last = runner.normalize_hidden_state(full.hidden_states[-1])[0, -1, :].to(runner.device).float()

        for method, patch_map in candidates.items():
            out = _run_low_prefill_with_patch(
                runner=runner,
                input_ids=input_ids,
                attention_mask=attention_mask,
                patch_map=patch_map,
            )
            method_logits = out.logits[:, -1, :].to(runner.device).float()
            method_hidden_last = runner.normalize_hidden_state(out.hidden_states[-1])[0, -1, :].to(runner.device).float()

            kl = float(F.kl_div(F.log_softmax(method_logits, dim=-1), full_probs, reduction="batchmean").item())
            top1 = float((method_logits.argmax(dim=-1) == full_logits.argmax(dim=-1)).float().mean().item())
            hcos = float(F.cosine_similarity(method_hidden_last.unsqueeze(0), full_hidden_last.unsqueeze(0), dim=-1).item())

            rows.append(
                {
                    "prompt_idx": i,
                    "method": method,
                    "kl_full_to_method": kl,
                    "top1_match_to_full": top1,
                    "final_hidden_cos_to_full": hcos,
                }
            )
            per_method[method]["kl_full_to_method"].append(kl)
            per_method[method]["top1_match_to_full"].append(top1)
            per_method[method]["final_hidden_cos_to_full"].append(hcos)

    with (out_dir / "diagnostic_rows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)

    summary_rows: list[dict[str, Any]] = []
    for method in methods:
        summary_rows.append(
            {
                "method": method,
                "mean_kl_full_to_method": _mean(per_method[method]["kl_full_to_method"]),
                "mean_top1_match_to_full": _mean(per_method[method]["top1_match_to_full"]),
                "mean_final_hidden_cos_to_full": _mean(per_method[method]["final_hidden_cos_to_full"]),
            }
        )
    summary = {
        "train_config": str(train_cfg_path),
        "checkpoint": str(checkpoint),
        "extract_config": str(extract_cfg_path),
        "num_prompts": len(prompts),
        "layer": layer,
        "rows": summary_rows,
    }
    (out_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = ["# Learned vs Reference Patch Diagnostics", ""]
    report.append(f"- prompts: {len(prompts)}")
    report.append(f"- layer: {layer}")
    report.append("")
    report.append("| method | mean KL(full||method) | top1 match to full | final hidden cosine to full |")
    report.append("|---|---:|---:|---:|")
    for row in summary_rows:
        report.append(
            f"| {row['method']} | {row['mean_kl_full_to_method']:.6f} | "
            f"{row['mean_top1_match_to_full']:.6f} | {row['mean_final_hidden_cos_to_full']:.6f} |"
        )
    (out_dir / "diagnostic_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "summary_rows": summary_rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
