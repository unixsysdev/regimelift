#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from helmas3n.src.data.paired_dataset import PairedStateDataset
from helmas3n.src.eval.handoff_metrics import mean_cosine_similarity, mean_squared_error, sparse_top1_agreement


def analyze_dataset(root_dir: Path) -> dict:
    ds = PairedStateDataset(root_dir=root_dir, target="residual")

    per_layer_mse = defaultdict(list)
    per_layer_cos = defaultdict(list)
    top1_list = []

    for i in range(len(ds)):
        s = ds[i]
        layer = int(s["layer"].item())
        low = s["source_state"].unsqueeze(0).float()
        full = s["target_state"].unsqueeze(0).float()

        per_layer_mse[layer].append(mean_squared_error(low, full).item())
        per_layer_cos[layer].append(mean_cosine_similarity(low, full).item())

        if "low_logits_indices" in s and "full_logits_indices" in s:
            top1_list.append(
                sparse_top1_agreement(
                    indices_a=s["low_logits_indices"].unsqueeze(0),
                    values_a=s["low_logits_values"].unsqueeze(0),
                    indices_b=s["full_logits_indices"].unsqueeze(0),
                    values_b=s["full_logits_values"].unsqueeze(0),
                ).item()
            )

    layers = sorted(per_layer_mse)
    per_layer = {
        str(layer): {
            "mse": float(sum(per_layer_mse[layer]) / len(per_layer_mse[layer])),
            "cosine": float(sum(per_layer_cos[layer]) / len(per_layer_cos[layer])),
        }
        for layer in layers
    }

    summary = {
        "num_samples": len(ds),
        "num_layers": len(layers),
        "global_top1_low_vs_full": float(sum(top1_list) / max(len(top1_list), 1)) if top1_list else 0.0,
        "per_layer": per_layer,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze low/full regime alignment for paired states")
    parser.add_argument("--config", type=str, required=True, help="Extraction config path")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    out_dir = Path(cfg["output"]["dir"])
    if not out_dir.is_absolute():
        out_dir = (config_path.parent / out_dir).resolve()
    summary = analyze_dataset(out_dir)

    report_path = out_dir / "alignment_report.json"
    report_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
