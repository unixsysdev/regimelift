#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from helmas3n.scripts.eval_handoff import evaluate, evaluate_live_handoff
from helmas3n.scripts.train_uplift import _load_yaml, train


DEFAULT_METHOD_CONFIGS = [
    "configs/train_residual_identity.yaml",
    "configs/train_residual_mean_delta.yaml",
    "configs/train_residual_global_linear.yaml",
    "configs/train_residual.yaml",  # expected MLP default
]


def _method_label(cfg: dict[str, Any]) -> str:
    name = cfg.get("experiment", {}).get("name", "")
    kind = cfg.get("model", {}).get("kind", "")
    if name:
        return name
    return kind or "unknown"


def run_pilot(
    config_paths: list[Path],
    extract_config: Path,
    out_dir: Path,
    eval_max_batches: int,
    live_max_prompts: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    curves = []

    for cfg_path in config_paths:
        cfg = _load_yaml(cfg_path)
        method = _method_label(cfg)
        print(f"[pilot] training {method} from {cfg_path}")
        summary = train(cfg)

        ckpt = Path(cfg["experiment"]["out_dir"]) / "best.pt"
        if not ckpt.exists():
            ckpt = Path(cfg["experiment"]["out_dir"]) / "last.pt"

        offline = evaluate(cfg=cfg, checkpoint_path=ckpt, max_batches=eval_max_batches)

        horizon_to_match = {}
        for horizon in [1, 4, 8, 16]:
            live = evaluate_live_handoff(
                train_cfg=cfg,
                checkpoint_path=ckpt,
                extract_cfg_path=extract_config,
                max_prompts=live_max_prompts,
                max_new_tokens=horizon,
            )
            horizon_to_match[horizon] = live
            curves.append(
                {
                    "method": method,
                    "horizon": horizon,
                    "match_uplift": live["short_continuation_match_rate_uplift_vs_full"],
                    "match_low": live["short_continuation_match_rate_low_vs_full"],
                }
            )

        row = {
            "method": method,
            "residual_cosine": offline["state_cosine"],
            "residual_mse": offline["state_mse"],
            "next_token_kl_to_full": offline.get("next_token_kl_uplift_vs_full"),
            "next_token_top1_to_full": offline.get("next_token_top1_uplift_vs_full"),
            "8tok_cont_match_to_full": horizon_to_match[8]["short_continuation_match_rate_uplift_vs_full"],
            "8tok_low_cont_match_to_full": horizon_to_match[8]["short_continuation_match_rate_low_vs_full"],
            "8tok_delta_over_low": horizon_to_match[8]["uplift_delta_over_low"],
            "checkpoint": str(ckpt),
            "final_val_mse": summary.get("final_val_mse"),
        }
        rows.append(row)

    table_df = pd.DataFrame(rows)
    curve_df = pd.DataFrame(curves)

    table_df.to_csv(out_dir / "pilot_table.csv", index=False)
    curve_df.to_csv(out_dir / "continuation_curves.csv", index=False)
    (out_dir / "pilot_table.json").write_text(json.dumps(rows, indent=2))

    plt.figure(figsize=(8, 4.5))
    for method in curve_df["method"].unique():
        sub = curve_df[curve_df["method"] == method].sort_values("horizon")
        plt.plot(sub["horizon"], sub["match_uplift"], marker="o", label=f"{method} uplift")
    plt.xlabel("Decode horizon")
    plt.ylabel("Continuation match vs full")
    plt.title("Uplift Continuation Match by Horizon")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "continuation_match_uplift.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    for method in curve_df["method"].unique():
        sub = curve_df[curve_df["method"] == method].sort_values("horizon")
        plt.plot(sub["horizon"], sub["match_low"], marker="o", linestyle="--", label=f"{method} low")
    plt.xlabel("Decode horizon")
    plt.ylabel("Low continuation match vs full")
    plt.title("Low-Regime Continuation Match by Horizon")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "continuation_match_low.png", dpi=160)
    plt.close()

    result = {
        "table_path": str(out_dir / "pilot_table.csv"),
        "curves_path": str(out_dir / "continuation_curves.csv"),
        "num_methods": len(table_df),
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RegimeLift residual pilot baseline suite")
    parser.add_argument("--extract-config", type=str, default="configs/extract.yaml")
    parser.add_argument("--methods", type=str, nargs="*", default=DEFAULT_METHOD_CONFIGS)
    parser.add_argument("--out-dir", type=str, default="artifacts/reports/pilot_residual")
    parser.add_argument("--eval-max-batches", type=int, default=50)
    parser.add_argument("--live-max-prompts", type=int, default=20)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    config_paths = []
    for rel in args.methods:
        p = Path(rel)
        if not p.is_absolute():
            p = (script_dir.parent / rel).resolve()
        config_paths.append(p)

    extract_cfg = Path(args.extract_config)
    if not extract_cfg.is_absolute():
        extract_cfg = (script_dir.parent / args.extract_config).resolve()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (script_dir.parent / args.out_dir).resolve()

    run_pilot(
        config_paths=config_paths,
        extract_config=extract_cfg,
        out_dir=out_dir,
        eval_max_batches=args.eval_max_batches,
        live_max_prompts=args.live_max_prompts,
    )


if __name__ == "__main__":
    main()
