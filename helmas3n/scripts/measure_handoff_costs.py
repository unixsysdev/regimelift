#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from helmas3n.scripts.train_uplift import _build_model, _load_yaml
from helmas3n.src.eval.decode_resume import resume_with_kv_cache, resume_with_residual_patch
from helmas3n.src.eval.handoff_metrics import continuation_match_rate
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


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak_mem_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated() / (1024 ** 2))


def _reset_peak_mem() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure low/full/handoff inference costs at fixed layer34,last1.")
    parser.add_argument("--train-config", type=str, default="configs/train_residual_layer34_last1.yaml")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--extract-config", type=str, default="configs/extract_killtest40_holdout80.yaml")
    parser.add_argument("--layer", type=int, default=34)
    parser.add_argument("--max-prompts", type=int, default=80)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--out-dir", type=str, default="artifacts/reports/cost_table_layer34_last1")
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

    rows: list[dict[str, Any]] = []
    low_prefill_ms: list[float] = []
    low_prefill_mem: list[float] = []
    full_restart_ms: list[float] = []
    full_restart_mem: list[float] = []
    handoff_resume_ms: list[float] = []
    handoff_resume_mem: list[float] = []
    pipeline_low_plus_handoff_ms: list[float] = []
    uplift_match_h16: list[float] = []
    low_match_h16: list[float] = []

    layer = int(args.layer)
    for i, prompt in enumerate(prompts, start=1):
        if i == 1 or i % 10 == 0 or i == len(prompts):
            print(f"[cost] prompt {i}/{len(prompts)}", flush=True)

        input_ids, attention_mask = runner.tokenize(prompt, max_length=extract_cfg.max_length)

        _reset_peak_mem()
        _sync()
        t0 = time.perf_counter()
        low = runner.forward_prefix(input_ids, attention_mask, regime="low", capture_kv=True, capture_logits=True)
        _sync()
        dt_low = (time.perf_counter() - t0) * 1000.0
        low_prefill_ms.append(dt_low)
        pm = _peak_mem_mb()
        if pm is not None:
            low_prefill_mem.append(pm)

        _reset_peak_mem()
        _sync()
        t0 = time.perf_counter()
        full = runner.forward_prefix(input_ids, attention_mask, regime="full", capture_kv=True, capture_logits=True)
        full_tokens = resume_with_kv_cache(
            runner=runner,
            uplifted_past_key_values=full.past_key_values,
            last_token_ids=input_ids[:, -1:],
            regime="full",
            max_new_tokens=args.max_new_tokens,
            seed_token_ids=full.logits[:, -1, :].argmax(dim=-1, keepdim=True),
        )
        _sync()
        dt_full_restart = (time.perf_counter() - t0) * 1000.0
        full_restart_ms.append(dt_full_restart)
        pm = _peak_mem_mb()
        if pm is not None:
            full_restart_mem.append(pm)

        low_last = runner.normalize_hidden_state(low.hidden_states[layer + 1])[0, -1, :].detach().to(runner.device).float().unsqueeze(0)
        with torch.no_grad():
            patch_state = model(low_last, torch.tensor([layer], device=runner.device)).squeeze(0).detach().cpu()

        _reset_peak_mem()
        _sync()
        t0 = time.perf_counter()
        uplift_tokens = resume_with_residual_patch(
            runner=runner,
            input_ids=input_ids,
            attention_mask=attention_mask,
            patch_map={layer: patch_state},
            regime="low",
            decode_regime="full",
            max_new_tokens=args.max_new_tokens,
        )
        _sync()
        dt_handoff = (time.perf_counter() - t0) * 1000.0
        handoff_resume_ms.append(dt_handoff)
        pm = _peak_mem_mb()
        if pm is not None:
            handoff_resume_mem.append(pm)

        low_native_tokens = resume_with_kv_cache(
            runner=runner,
            uplifted_past_key_values=low.past_key_values,
            last_token_ids=input_ids[:, -1:],
            regime="low",
            max_new_tokens=args.max_new_tokens,
            seed_token_ids=low.logits[:, -1, :].argmax(dim=-1, keepdim=True),
        )

        pipeline_low_plus_handoff_ms.append(dt_low + dt_handoff)
        uplift_match_h16.append(float(continuation_match_rate(full_tokens.cpu(), uplift_tokens.cpu()).item()))
        low_match_h16.append(float(continuation_match_rate(full_tokens.cpu(), low_native_tokens.cpu()).item()))

        rows.append(
            {
                "prompt_idx": i,
                "low_prefill_ms": dt_low,
                "full_restart_ms": dt_full_restart,
                "handoff_resume_ms": dt_handoff,
                "pipeline_low_plus_handoff_ms": dt_low + dt_handoff,
                "uplift_match_h16": uplift_match_h16[-1],
                "low_match_h16": low_match_h16[-1],
            }
        )

    with (out_dir / "cost_rows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)

    summary = {
        "train_config": str(train_cfg_path),
        "checkpoint": str(checkpoint),
        "extract_config": str(extract_cfg_path),
        "num_prompts": len(prompts),
        "max_new_tokens": int(args.max_new_tokens),
        "mean_low_prefill_ms": _mean(low_prefill_ms),
        "mean_full_restart_ms": _mean(full_restart_ms),
        "mean_handoff_resume_ms": _mean(handoff_resume_ms),
        "mean_pipeline_low_plus_handoff_ms": _mean(pipeline_low_plus_handoff_ms),
        "delta_pipeline_vs_full_restart_ms": _mean(pipeline_low_plus_handoff_ms) - _mean(full_restart_ms),
        "mean_uplift_match_h16": _mean(uplift_match_h16),
        "mean_low_match_h16": _mean(low_match_h16),
        "mean_low_prefill_mem_mb": _mean(low_prefill_mem) if low_prefill_mem else None,
        "mean_full_restart_mem_mb": _mean(full_restart_mem) if full_restart_mem else None,
        "mean_handoff_resume_mem_mb": _mean(handoff_resume_mem) if handoff_resume_mem else None,
    }
    (out_dir / "cost_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "# Cost Table @ layer34,last1",
        "",
        f"- prompts: {summary['num_prompts']}",
        f"- decode horizon: {summary['max_new_tokens']}",
        "",
        "| Component | Mean ms/prompt |",
        "|---|---:|",
        f"| low prefill | {summary['mean_low_prefill_ms']:.3f} |",
        f"| full restart (prefill+decode) | {summary['mean_full_restart_ms']:.3f} |",
        f"| handoff resume (patched prefill+decode) | {summary['mean_handoff_resume_ms']:.3f} |",
        f"| pipeline: low prefill + handoff resume | {summary['mean_pipeline_low_plus_handoff_ms']:.3f} |",
        "",
        f"- pipeline minus full restart: {summary['delta_pipeline_vs_full_restart_ms']:.3f} ms",
        f"- h16 continuation match: uplift={summary['mean_uplift_match_h16']:.6f}, low={summary['mean_low_match_h16']:.6f}",
    ]
    (out_dir / "cost_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
