#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
import sys
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from helmas3n.src.data.paired_dataset import PairedStateDataset


def _load_source_dataset(root_dir: Path) -> PairedStateDataset:
    return PairedStateDataset(str(root_dir), target="residual", shard_cache_size=8)


def _prompt_max_positions(dataset: PairedStateDataset, target_layer: int) -> OrderedDict[str, int]:
    max_positions: OrderedDict[str, int] = OrderedDict()
    for idx in range(len(dataset)):
        sample = dataset[idx]
        layer = int(sample["layer"].item())
        if layer != target_layer:
            continue
        prompt_id = str(sample["prompt_id"])
        token_position = int(sample["token_position"].item())
        prev = max_positions.get(prompt_id)
        if prev is None or token_position > prev:
            max_positions[prompt_id] = token_position
    return max_positions


def _select_indices(dataset: PairedStateDataset, target_layer: int, max_positions: dict[str, int]) -> list[int]:
    selected: list[int] = []
    seen: set[tuple[str, int, int]] = set()
    for idx in range(len(dataset)):
        sample = dataset[idx]
        layer = int(sample["layer"].item())
        if layer != target_layer:
            continue
        prompt_id = str(sample["prompt_id"])
        token_position = int(sample["token_position"].item())
        if token_position != max_positions.get(prompt_id):
            continue
        key = (prompt_id, layer, token_position)
        if key in seen:
            continue
        seen.add(key)
        selected.append(idx)
    return selected


def _gather_sample(dataset: PairedStateDataset, index: int) -> dict[str, Any]:
    sample = dataset[index]
    out: dict[str, Any] = {
        "prompt_id": sample["prompt_id"],
        "layer": sample["layer"].clone(),
        "token_position": sample["token_position"].clone(),
        "residual_low": sample["source_state"].clone(),
        "residual_full": sample["target_state"].clone(),
    }
    for key in [
        "k_low",
        "v_low",
        "k_full",
        "v_full",
        "full_logits_values",
        "full_logits_indices",
        "low_logits_values",
        "low_logits_indices",
        "low_on_full_logits_values",
    ]:
        if key in sample:
            value = sample[key]
            out[key] = value.clone() if torch.is_tensor(value) else value
    return out


def _write_targeted_root(
    source_root: Path,
    out_root: Path,
    target_layer: int,
    repeat_to_match_source: bool,
) -> dict[str, Any]:
    dataset = _load_source_dataset(source_root)
    max_positions = _prompt_max_positions(dataset, target_layer=target_layer)
    selected_indices = _select_indices(dataset, target_layer=target_layer, max_positions=max_positions)
    if not selected_indices:
        raise RuntimeError(
            f"No samples matched layer={target_layer} in {source_root}. "
            "Check that the source paired states contain the requested layer."
        )

    selected_samples = [_gather_sample(dataset, idx) for idx in selected_indices]
    source_count = len(dataset)
    repeat_factor = 1
    repeated_samples = selected_samples
    if repeat_to_match_source and len(selected_samples) < source_count:
        repeat_factor = (source_count + len(selected_samples) - 1) // len(selected_samples)
        repeated_samples = (selected_samples * repeat_factor)[:source_count]

    out_root.mkdir(parents=True, exist_ok=True)
    shard_path = out_root / "shard_00000.pt"

    shard: dict[str, Any] = {}
    shard["prompt_id"] = [str(sample["prompt_id"]) for sample in repeated_samples]
    shard["layer"] = torch.stack([sample["layer"] for sample in repeated_samples], dim=0)
    shard["token_position"] = torch.stack([sample["token_position"] for sample in repeated_samples], dim=0)
    shard["residual_low"] = torch.stack([sample["residual_low"] for sample in repeated_samples], dim=0)
    shard["residual_full"] = torch.stack([sample["residual_full"] for sample in repeated_samples], dim=0)

    optional_keys = [
        "k_low",
        "v_low",
        "k_full",
        "v_full",
        "full_logits_values",
        "full_logits_indices",
        "low_logits_values",
        "low_logits_indices",
        "low_on_full_logits_values",
    ]
    for key in optional_keys:
        if key in repeated_samples[0]:
            shard[key] = torch.stack([sample[key] for sample in repeated_samples], dim=0)

    torch.save(shard, shard_path)

    manifest = {
        "format": "paired_states_subset",
        "num_samples": source_count if repeat_to_match_source else len(selected_samples),
        "shards": [{"path": shard_path.name, "num_samples": source_count if repeat_to_match_source else len(selected_samples)}],
        "capture_kv": bool(dataset.info.capture_kv),
        "capture_logits": bool(dataset.info.capture_logits),
        "logits_topk": int(dataset.info.logits_topk),
        "num_layers": int(dataset.info.num_layers),
        "hidden_dim": int(dataset.info.hidden_dim),
        "num_heads": int(dataset.manifest.get("num_heads", 0)),
        "head_dim": int(dataset.manifest.get("head_dim", 0)),
        "model_name": dataset.manifest.get("model_name"),
        "selection": {
            "target_layer": target_layer,
            "position_mode": "last1",
            "unique_samples": len(selected_samples),
            "repeat_to_match_source": repeat_to_match_source,
            "repeat_factor": repeat_factor,
            "source_root": str(source_root),
            "source_num_samples": source_count,
        },
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_root / "selection.json").write_text(
        json.dumps(
            {
                "target_layer": target_layer,
                "position_mode": "last1",
                "selected_prompt_ids": list(max_positions.keys()),
                "unique_sample_count": len(selected_samples),
                "repeat_to_match_source": repeat_to_match_source,
                "repeat_factor": repeat_factor,
                "source_root": str(source_root),
                "source_num_samples": source_count,
                "out_root": str(out_root),
            },
            indent=2,
        )
    )

    return {
        "source_root": str(source_root),
        "out_root": str(out_root),
        "target_layer": target_layer,
        "unique_samples": len(selected_samples),
        "output_samples": source_count if repeat_to_match_source else len(selected_samples),
        "repeat_factor": repeat_factor,
        "selected_prompt_ids": list(max_positions.keys()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a targeted paired-state subset for a single layer/last1 target")
    parser.add_argument("--source-root", type=str, required=True)
    parser.add_argument("--target-layer", type=int, required=True)
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--no-repeat", action="store_true", help="Keep only the unique selected samples instead of repeating to source length")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    out_root = Path(args.out_root).resolve()
    summary = _write_targeted_root(
        source_root=source_root,
        out_root=out_root,
        target_layer=int(args.target_layer),
        repeat_to_match_source=not bool(args.no_repeat),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
