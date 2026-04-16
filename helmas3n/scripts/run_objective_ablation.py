#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from helmas3n.scripts.eval_handoff import evaluate, evaluate_live_handoff
from helmas3n.scripts.train_uplift import _load_yaml, train


DEFAULT_CONFIGS = [
    "configs/train_residual_layer34_last1_state_only.yaml",
    "configs/train_residual_layer34_last1_state_logit.yaml",
    "configs/train_residual_layer34_last1_heavy_logit.yaml",
    "configs/train_residual_layer34_last1_short_horizon.yaml",
]


def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = (PROJECT_ROOT / "helmas3n" / p).resolve()
    return p


def _parse_horizons(value: str) -> list[int]:
    out = sorted({int(x.strip()) for x in value.split(",") if x.strip()})
    if not out:
        raise ValueError("At least one horizon is required.")
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_reference_metrics(path: Path) -> dict[str, float]:
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    ref_rows = [r for r in rows if r.get("split") == "heldout" and r.get("experiment_site") == "layer34"]
    if not ref_rows:
        raise ValueError(f"No heldout/layer34 rows in reference CSV: {path}")
    no_patch = next(r for r in ref_rows if r["method"] == "low_to_full_no_patch")
    ref34 = next(r for r in ref_rows if r["method"] == "oracle_layer34")
    baseline = next(r for r in ref_rows if r["method"] == "targeted_mlp")
    out: dict[str, float] = {}
    for h in [1, 4, 8, 16]:
        out[f"no_patch_h{h}"] = float(no_patch[f"cont_match_h{h}_uplift_vs_full"])
        out[f"reference_h{h}"] = float(ref34[f"cont_match_h{h}_uplift_vs_full"])
        out[f"baseline_h{h}"] = float(baseline[f"cont_match_h{h}_uplift_vs_full"])
    return out


def _write_report(path: Path, rows: list[dict[str, Any]], horizons: list[int]) -> None:
    def _fmt(v: Any) -> str:
        if v is None:
            return "--"
        return f"{float(v):.3f}"

    lines: list[str] = []
    lines.append("# Objective Ablation @ layer34,last1")
    lines.append("")
    lines.append(f"- Variants: {len(rows)}")
    lines.append(f"- Horizons: {', '.join(f'h{h}' for h in horizons)}")
    lines.append("")
    lines.append("| method | train loss mix | held-out h1 | held-out h4 | held-out h8 | held-out h16 | delta_h8 | delta_h16 | cost per prompt (ms, h16) | closure to reference-layer34 (h8) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['variant']} | `{row['train_loss_mix']}` | "
            f"{row.get('h1_uplift', 0.0):.6f} | {row.get('h4_uplift', 0.0):.6f} | "
            f"{row.get('h8_uplift', 0.0):.6f} | {row.get('h16_uplift', 0.0):.6f} | "
            f"{row.get('delta_h8', 0.0):.6f} | {row.get('delta_h16', 0.0):.6f} | "
            f"{_fmt(row.get('cost_per_prompt_ms_h16'))} | {_fmt(row.get('closure_reference_h8'))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-site objective ablation at layer34,last1.")
    parser.add_argument(
        "--train-configs",
        type=str,
        default=",".join(DEFAULT_CONFIGS),
        help="Comma-separated config paths relative to helmas3n/ or absolute.",
    )
    parser.add_argument(
        "--extract-config",
        type=str,
        default="configs/extract_killtest40_holdout80.yaml",
        help="Extractor config used for live heldout evaluation.",
    )
    parser.add_argument(
        "--reference-rows",
        type=str,
        default="artifacts/reports/targeted_site_study_v5_holdout80/targeted_site_rows.csv",
        help="Reference heldout rows CSV (used for closure-to-reference-layer34).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/reports/objective_ablation_layer34_last1",
        help="Output directory relative to helmas3n/ or absolute.",
    )
    parser.add_argument("--horizons", type=str, default="1,4,8,16")
    parser.add_argument("--max-prompts", type=int, default=80)
    parser.add_argument("--max-eval-batches", type=int, default=50)
    parser.add_argument("--reuse-checkpoints", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing rows CSV in out-dir if present.")
    args = parser.parse_args()

    train_configs = [_resolve(x.strip()) for x in args.train_configs.split(",") if x.strip()]
    extract_cfg = _resolve(args.extract_config)
    reference_rows = _resolve(args.reference_rows)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    horizons = _parse_horizons(args.horizons)
    reference = _load_reference_metrics(reference_rows)

    rows_path = out_dir / "objective_ablation_rows.csv"
    existing_rows = _load_existing_rows(rows_path) if args.resume else []
    rows: list[dict[str, Any]] = list(existing_rows)
    done_variants = {str(row.get("variant", "")) for row in existing_rows}

    for cfg_path in train_configs:
        cfg = _load_yaml(cfg_path)
        variant = cfg["experiment"]["name"]
        ckpt_dir = Path(cfg["experiment"]["out_dir"])
        checkpoint = ckpt_dir / "last.pt"

        if variant in done_variants:
            print(f"[ablation] resume skip completed variant: {variant}", flush=True)
            continue

        if args.reuse_checkpoints and checkpoint.exists():
            print(f"[ablation] reuse checkpoint for {variant}: {checkpoint}", flush=True)
        else:
            print(f"[ablation] train variant: {variant}", flush=True)
            train(cfg)

        print(f"[ablation] evaluate local metrics: {variant}", flush=True)
        local_metrics = evaluate(cfg=cfg, checkpoint_path=checkpoint, max_batches=args.max_eval_batches)

        row: dict[str, Any] = {
            "variant": variant,
            "config_path": str(cfg_path),
            "checkpoint_path": str(checkpoint),
            "train_loss_mix": (
                f"mse={cfg['loss'].get('mse_weight', 0)};"
                f"cos={cfg['loss'].get('cosine_weight', 0)};"
                f"kl={cfg['loss'].get('logit_weight', 0)};"
                f"ce={cfg['loss'].get('next_token_ce_weight', 0)}"
            ),
            "val_mse": float(local_metrics["state_mse"]),
            "val_cosine": float(local_metrics["state_cosine"]),
            "next_token_top1_low_vs_full": float(local_metrics["next_token_top1_low_vs_full"]),
            "next_token_top1_uplift_vs_full": float(local_metrics["next_token_top1_uplift_vs_full"]),
            "next_token_kl_low_vs_full": local_metrics["next_token_kl_low_vs_full"],
            "next_token_kl_uplift_vs_full": local_metrics["next_token_kl_uplift_vs_full"],
        }

        for h in horizons:
            print(f"[ablation] live handoff {variant} @ h{h}", flush=True)
            t0 = time.perf_counter()
            live = evaluate_live_handoff(
                train_cfg=cfg,
                checkpoint_path=checkpoint,
                extract_cfg_path=extract_cfg,
                max_prompts=args.max_prompts,
                max_new_tokens=h,
            )
            dt = time.perf_counter() - t0
            uplift = float(live["short_continuation_match_rate_uplift_vs_full"])
            no_patch = float(live["short_continuation_match_rate_low_vs_full"])
            row[f"h{h}_uplift"] = uplift
            row[f"h{h}_no_patch"] = no_patch
            row[f"delta_h{h}"] = uplift - no_patch
            next_top1 = live.get("next_token_top1_uplift_vs_full_live")
            if next_top1 is None:
                next_top1 = live.get("next_token_top1_uplift_vs_full_proxy", 0.0)
            row[f"h{h}_next_top1_uplift_vs_full"] = float(next_top1)
            row[f"cost_per_prompt_ms_h{h}"] = (dt * 1000.0) / max(args.max_prompts, 1)

        for h in [8, 16]:
            if f"delta_h{h}" not in row:
                row[f"closure_reference_h{h}"] = None
                row[f"delta_vs_baseline_h{h}"] = None
                continue
            ref_gap = reference[f"reference_h{h}"] - reference[f"no_patch_h{h}"]
            if abs(ref_gap) > 1e-12:
                row[f"closure_reference_h{h}"] = row[f"delta_h{h}"] / ref_gap
            else:
                row[f"closure_reference_h{h}"] = None
            row[f"delta_vs_baseline_h{h}"] = row[f"h{h}_uplift"] - reference[f"baseline_h{h}"]

        rows.append(row)
        summary = {
            "extract_config": str(extract_cfg),
            "reference_rows": str(reference_rows),
            "max_prompts": args.max_prompts,
            "horizons": horizons,
            "rows": rows,
            "best_by_delta_h8": max(rows, key=lambda r: float(r.get("delta_h8", -1e9)))["variant"] if rows else None,
            "best_by_delta_h16": max(rows, key=lambda r: float(r.get("delta_h16", -1e9)))["variant"] if rows else None,
        }
        _write_csv(rows_path, rows)
        _write_report(out_dir / "objective_ablation_report.md", rows, horizons)
        _write_summary(out_dir / "objective_ablation_summary.json", summary)

    print(json.dumps({"out_dir": str(out_dir), "num_rows": len(rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
