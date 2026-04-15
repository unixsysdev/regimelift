#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Callable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from helmas3n.scripts.run_layer_sweep import (
    PromptCacheItem,
    _aggregate_baseline,
    _build_prompt_cache,
    _load_prompts_from_extract_cfg,
    _set_seed,
)
from helmas3n.scripts.train_uplift import _build_model, _load_yaml
from helmas3n.src.data.paired_dataset import PairedStateDataset
from helmas3n.src.eval.decode_resume import resume_with_residual_patch
from helmas3n.src.eval.handoff_metrics import continuation_match_rate
from helmas3n.src.gemma.runner import GemmaRunner
from helmas3n.src.gemma.state_extract import load_extraction_config


EXPERIMENT_SITES: dict[str, int] = {
    "layer16": 16,
    "layer34": 34,
}
STRIDE_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 34]
HORIZONS = [1, 4, 8, 16]


def _closure(method: float, no_patch: float, oracle: float) -> float:
    denom = oracle - no_patch
    if abs(denom) < 1e-9:
        return 0.0
    return (method - no_patch) / denom


def _resolve_absolute_path(raw: str, label: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path: {raw}")
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _load_prompt_records(extract_cfg_path: Path, max_prompts: int) -> tuple[list[dict[str, str]], Path]:
    cfg = load_extraction_config(extract_cfg_path)
    prompts_path = Path(cfg.prompts_path)
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompt file does not exist: {prompts_path}")

    prompt_field = cfg.prompt_field
    id_field = cfg.id_field
    records: list[dict[str, str]] = []

    if prompts_path.suffix == ".jsonl":
        for line in prompts_path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            records.append(
                {
                    "id": str(item.get(id_field, item.get("id", len(records)))),
                    "prompt": str(item.get(prompt_field, item.get("prompt", ""))),
                }
            )
    elif prompts_path.suffix == ".json":
        raw = json.loads(prompts_path.read_text())
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    records.append(
                        {
                            "id": str(item.get(id_field, item.get("id", len(records)))),
                            "prompt": str(item.get(prompt_field, item.get("prompt", ""))),
                        }
                    )
                else:
                    records.append({"id": str(len(records)), "prompt": str(item)})
        else:
            raise ValueError(f"Unsupported prompt file format: {prompts_path}")
    else:
        for i, line in enumerate(prompts_path.read_text().splitlines()):
            if line.strip():
                records.append({"id": str(i), "prompt": line.strip()})

    return records[:max_prompts], prompts_path


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


def _write_run_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def _flush_artifacts(
    out_dir: Path,
    rows: list[dict[str, Any]],
    horizons: list[int],
    run_manifest: dict[str, Any],
    split_row_counts: dict[str, int],
    phase: str,
) -> None:
    _write_csv(out_dir / "targeted_site_rows.csv", rows)
    _write_run_manifest(
        out_dir / "targeted_site_summary.json",
        {
            **run_manifest,
            "phase": phase,
            "num_rows": len(rows),
            "split_row_counts": split_row_counts,
            "horizons": horizons,
        },
    )
    _write_report(out_dir / "targeted_site_report.md", rows, horizons)


def _load_model_from_checkpoint(checkpoint_path: Path, train_cfg: dict[str, Any], paired_root: Path) -> torch.nn.Module:
    dataset = PairedStateDataset(str(paired_root), target="residual")
    sample = dataset[0]
    hidden_dim = int(sample["source_state"].numel())
    num_layers = max(int(dataset.info.num_layers), int(sample["layer"].item()) + 1)
    model = _build_model(
        kind=train_cfg["model"]["kind"],
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        model_cfg=train_cfg["model"],
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _predict_patch(
    model: torch.nn.Module,
    layer: int,
    state: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    with torch.no_grad():
        pred = model(state.to(device).float().unsqueeze(0), torch.tensor([layer], device=device))
    return pred.squeeze(0).detach().cpu()


def _run_patch_method(
    runner: GemmaRunner,
    cache: list[PromptCacheItem],
    horizons: list[int],
    method_name: str,
    patch_builder: Callable[[PromptCacheItem], dict[int, torch.Tensor]],
) -> dict[str, Any]:
    max_h = max(horizons)
    total = len(cache)
    sums = {h: 0.0 for h in horizons}
    next_top1 = 0.0

    for i, item in enumerate(cache, start=1):
        if i == 1 or i % 10 == 0 or i == total:
            print(f"[targeted] {method_name}: prompt {i}/{total}", flush=True)
        patch_map = patch_builder(item)
        _set_seed(item.seed)
        uplift = resume_with_residual_patch(
            runner=runner,
            input_ids=item.input_ids.to(runner.device),
            attention_mask=item.attention_mask.to(runner.device),
            patch_map=patch_map,
            regime="low",
            decode_regime="full",
            max_new_tokens=max_h,
        ).cpu()
        for h in horizons:
            sums[h] += float(continuation_match_rate(item.full_tokens[:, :h], uplift[:, :h]).item())
        if uplift.numel() > 0 and item.full_tokens.numel() > 0:
            next_top1 += float((uplift[:, 0] == item.full_tokens[:, 0]).float().mean().item())

    row: dict[str, Any] = {
        "method": method_name,
        "prompt_count": total,
        "next_token_top1_uplift_vs_full_live": next_top1 / max(total, 1),
    }
    for h in horizons:
        row[f"cont_match_h{h}_uplift_vs_full"] = sums[h] / max(total, 1)
    return row


def _evaluate_block(
    runner: GemmaRunner,
    cache: list[PromptCacheItem],
    horizons: list[int],
    split_name: str,
    experiment_site: str,
    broad_model: torch.nn.Module,
    targeted_model: torch.nn.Module,
    baseline: dict[str, float],
) -> list[dict[str, Any]]:
    site_layer = EXPERIMENT_SITES[experiment_site]

    methods: list[tuple[str, Callable[[PromptCacheItem], dict[int, torch.Tensor]]]] = [
        ("low_to_full_no_patch", lambda item: {}),
        ("identity", lambda item: {site_layer: item.low_states[site_layer]}),
        (
            "broad_mlp",
            lambda item: {site_layer: _predict_patch(broad_model, site_layer, item.low_states[site_layer], runner.device)},
        ),
        (
            "targeted_mlp",
            lambda item: {site_layer: _predict_patch(targeted_model, site_layer, item.low_states[site_layer], runner.device)},
        ),
        ("oracle_layer16", lambda item: {16: item.full_states[16]}),
        ("oracle_layer34", lambda item: {34: item.full_states[34]}),
        ("oracle_stride", lambda item: {layer: item.full_states[layer] for layer in STRIDE_LAYERS}),
    ]

    rows: list[dict[str, Any]] = []
    for method_name, patch_builder in methods:
        row = _run_patch_method(
            runner=runner,
            cache=cache,
            horizons=horizons,
            method_name=method_name,
            patch_builder=patch_builder,
        )
        row["split"] = split_name
        row["experiment_site"] = experiment_site
        row["site_layer"] = site_layer
        rows.append(row)

    lookup = {str(row["method"]): row for row in rows}
    same_site_oracle = lookup[f"oracle_{experiment_site}"]
    stride_oracle = lookup["oracle_stride"]

    for row in rows:
        for h in horizons:
            no_patch = baseline[f"low_to_full_live_nopatch_h{h}"]
            same_site_oracle_match = same_site_oracle[f"cont_match_h{h}_uplift_vs_full"]
            stride_oracle_match = stride_oracle[f"cont_match_h{h}_uplift_vs_full"]

            row[f"no_patch_h{h}"] = no_patch
            row[f"same_site_oracle_h{h}"] = same_site_oracle_match
            row[f"stride_oracle_h{h}"] = stride_oracle_match
            row[f"delta_h{h}"] = row[f"cont_match_h{h}_uplift_vs_full"] - no_patch
            row[f"closure_same_site_h{h}"] = _closure(
                method=row[f"cont_match_h{h}_uplift_vs_full"],
                no_patch=no_patch,
                oracle=same_site_oracle_match,
            )
            row[f"closure_stride_h{h}"] = _closure(
                method=row[f"cont_match_h{h}_uplift_vs_full"],
                no_patch=no_patch,
                oracle=stride_oracle_match,
            )

    return rows


def _write_report(path: Path, rows: list[dict[str, Any]], horizons: list[int]) -> None:
    lines: list[str] = []
    lines.append("# Targeted Site Study")
    lines.append("")
    lines.append(f"- Rows: {len(rows)}")
    lines.append(f"- Horizons: {', '.join(f'h{h}' for h in horizons)}")
    lines.append("")

    for split in ["pilot", "heldout"]:
        split_rows = [r for r in rows if r["split"] == split]
        if not split_rows:
            continue
        lines.append(f"## {split}")
        lines.append("| Experiment site | Method | h8 | h16 | delta_h8 | delta_h16 | closure_same_site_h8 | closure_stride_h8 |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for row in split_rows:
            lines.append(
                f"| {row['experiment_site']} | {row['method']} | "
                f"{float(row['cont_match_h8_uplift_vs_full']):.6f} | {float(row['cont_match_h16_uplift_vs_full']):.6f} | "
                f"{float(row['delta_h8']):.6f} | {float(row['delta_h16']):.6f} | "
                f"{float(row['closure_same_site_h8']):.3f} | {float(row['closure_stride_h8']):.3f} |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n")


def _load_split_prompts(
    extract_path: Path,
    max_prompts: int,
    split_name: str,
    expected_prompts: int | None = None,
) -> tuple[list[dict[str, str]], Path]:
    records, prompt_path = _load_prompt_records(extract_path, max_prompts=max_prompts)
    count = len(records)
    if expected_prompts is not None and count != expected_prompts:
        raise RuntimeError(
            f"{split_name} prompt count mismatch: expected {expected_prompts}, loaded {count} from {prompt_path}"
        )
    if count != max_prompts:
        raise RuntimeError(
            f"{split_name} prompt count mismatch: expected {max_prompts}, loaded {count} from {prompt_path}"
        )
    first_id = records[0]["id"] if records else "<none>"
    last_id = records[-1]["id"] if records else "<none>"
    print(
        json.dumps(
            {
                "split": split_name,
                "resolved_extract_config": str(extract_path),
                "resolved_prompts_path": str(prompt_path),
                "loaded_prompt_count": count,
                "first_prompt_id": first_id,
                "last_prompt_id": last_id,
            },
            indent=2,
        ),
        flush=True,
    )
    return records, prompt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the targeted layer16/layer34 last1 site study")
    parser.add_argument("--pilot-extract-config", type=str, default="/home/marcel/Work/He-LMAS/helmas3n/configs/extract_killtest40.yaml")
    parser.add_argument("--heldout-extract-config", type=str, default="/home/marcel/Work/He-LMAS/helmas3n/configs/extract_killtest40_holdout80.yaml")
    parser.add_argument("--train-template", type=str, default="/home/marcel/Work/He-LMAS/helmas3n/configs/train_residual.yaml")
    parser.add_argument("--paired-root", type=str, default="/home/marcel/Work/He-LMAS/helmas3n/artifacts/paired_states_killtest40")
    parser.add_argument("--broad-checkpoint", type=str, default="/home/marcel/Work/He-LMAS/helmas3n/artifacts/reports/killtest_v3/checkpoints/mlp/last.pt")
    parser.add_argument("--layer16-checkpoint", type=str, default="/home/marcel/Work/He-LMAS/helmas3n/artifacts/reports/targeted_runs/checkpoints/layer16_last1/last.pt")
    parser.add_argument("--layer34-checkpoint", type=str, default="/home/marcel/Work/He-LMAS/helmas3n/artifacts/reports/targeted_runs/checkpoints/layer34_last1/last.pt")
    parser.add_argument("--out-dir", type=str, default="/home/marcel/Work/He-LMAS/helmas3n/artifacts/reports/targeted_site_study_v3")
    parser.add_argument("--max-prompts-pilot", type=int, default=40)
    parser.add_argument("--max-prompts-heldout", type=int, default=80)
    parser.add_argument("--expected-heldout-prompts", type=int, default=80)
    parser.add_argument("--heldout-only", action="store_true", help="Skip the pilot split and evaluate held-out only")
    parser.add_argument("--horizons", type=str, default="1,4,8,16")
    args = parser.parse_args()

    pilot_extract_path = _resolve_absolute_path(args.pilot_extract_config, "--pilot-extract-config")
    heldout_extract_path = _resolve_absolute_path(args.heldout_extract_config, "--heldout-extract-config")
    train_template_path = _resolve_absolute_path(args.train_template, "--train-template")
    paired_root = _resolve_absolute_path(args.paired_root, "--paired-root")
    broad_checkpoint = _resolve_absolute_path(args.broad_checkpoint, "--broad-checkpoint")
    layer16_checkpoint = _resolve_absolute_path(args.layer16_checkpoint, "--layer16-checkpoint")
    layer34_checkpoint = _resolve_absolute_path(args.layer34_checkpoint, "--layer34-checkpoint")
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        raise ValueError(f"--out-dir must be an absolute path: {args.out_dir}")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    horizons = sorted(set(horizons))
    if not horizons:
        raise ValueError("No horizons provided")

    train_cfg = _load_yaml(train_template_path)
    base_dataset = PairedStateDataset(str(paired_root), target="residual")
    sample = base_dataset[0]
    hidden_dim = int(sample["source_state"].numel())
    num_layers = max(int(base_dataset.info.num_layers), int(sample["layer"].item()) + 1)

    heldout_records, heldout_prompts_path = _load_split_prompts(
        heldout_extract_path,
        max_prompts=args.max_prompts_heldout,
        split_name="heldout",
        expected_prompts=args.expected_heldout_prompts,
    )

    pilot_records: list[dict[str, str]] = []
    pilot_prompts_path: Path | None = None
    if not args.heldout_only:
        pilot_records, pilot_prompts_path = _load_split_prompts(
            pilot_extract_path,
            max_prompts=args.max_prompts_pilot,
            split_name="pilot",
            expected_prompts=args.max_prompts_pilot,
        )

    runner_cfg = load_extraction_config(heldout_extract_path)
    runner = GemmaRunner(
        model_name=runner_cfg.model_name,
        device=runner_cfg.device,
        dtype=runner_cfg.dtype,
        trust_remote_code=runner_cfg.trust_remote_code,
        regimes=runner_cfg.regimes,
        model_load_overrides=runner_cfg.model_load_overrides,
        text_only=runner_cfg.text_only,
    )

    print(
        json.dumps(
            {
                "resolved_pilot_extract_config": str(pilot_extract_path),
                "resolved_heldout_extract_config": str(heldout_extract_path),
                "resolved_pilot_prompts_path": str(pilot_prompts_path) if pilot_prompts_path is not None else None,
                "resolved_heldout_prompts_path": str(heldout_prompts_path),
                "resolved_train_template": str(train_template_path),
                "resolved_paired_root": str(paired_root),
                "resolved_out_dir": str(out_dir),
                "broad_checkpoint": str(broad_checkpoint),
                "layer16_checkpoint": str(layer16_checkpoint),
                "layer34_checkpoint": str(layer34_checkpoint),
                "heldout_prompt_count": len(heldout_records),
                "heldout_prompt_id_first": heldout_records[0]["id"] if heldout_records else None,
                "heldout_prompt_id_last": heldout_records[-1]["id"] if heldout_records else None,
                "heldout_only": bool(args.heldout_only),
                "horizons": horizons,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
            },
            indent=2,
        ),
        flush=True,
    )

    run_manifest = {
        "resolved_pilot_extract_config": str(pilot_extract_path),
        "resolved_heldout_extract_config": str(heldout_extract_path),
        "resolved_pilot_prompts_path": str(pilot_prompts_path) if pilot_prompts_path is not None else None,
        "resolved_heldout_prompts_path": str(heldout_prompts_path),
        "resolved_train_template": str(train_template_path),
        "resolved_paired_root": str(paired_root),
        "resolved_out_dir": str(out_dir),
        "broad_checkpoint": str(broad_checkpoint),
        "layer16_checkpoint": str(layer16_checkpoint),
        "layer34_checkpoint": str(layer34_checkpoint),
        "heldout_prompt_count": len(heldout_records),
        "heldout_prompt_ids": [r["id"] for r in heldout_records],
        "heldout_only": bool(args.heldout_only),
        "horizons": horizons,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "max_prompts_pilot": args.max_prompts_pilot,
        "max_prompts_heldout": args.max_prompts_heldout,
        "expected_heldout_prompts": args.expected_heldout_prompts,
    }
    _write_run_manifest(out_dir / "run_manifest.json", run_manifest)

    broad_model = _load_model_from_checkpoint(broad_checkpoint, train_cfg=train_cfg, paired_root=paired_root)
    layer16_model = _load_model_from_checkpoint(layer16_checkpoint, train_cfg=train_cfg, paired_root=paired_root)
    layer34_model = _load_model_from_checkpoint(layer34_checkpoint, train_cfg=train_cfg, paired_root=paired_root)
    broad_model.to(runner.device)
    layer16_model.to(runner.device)
    layer34_model.to(runner.device)

    rows: list[dict[str, Any]] = []
    split_row_counts: dict[str, int] = {"pilot": 0, "heldout": 0}
    phase = "startup"

    def persist(row: dict[str, Any]) -> None:
        rows.append(row)
        split_row_counts[row["split"]] += 1
        _flush_artifacts(out_dir, rows, horizons, run_manifest, split_row_counts, phase)

    split_specs: list[tuple[str, Path, int, list[dict[str, str]]]] = []
    if not args.heldout_only:
        split_specs.append(("pilot", pilot_extract_path, args.max_prompts_pilot, pilot_records))
    split_specs.append(("heldout", heldout_extract_path, args.max_prompts_heldout, heldout_records))

    for split_name, extract_path, max_prompts, records in split_specs:
        print(f"[targeted] split={split_name} prompts={max_prompts}", flush=True)
        prompts = [r["prompt"] for r in records]
        cache = _build_prompt_cache(
            runner=runner,
            prompts=prompts,
            max_length=runner_cfg.max_length,
            max_h=max(horizons),
            base_seed=runner_cfg.seed,
        )
        baseline = _aggregate_baseline(runner=runner, cache=cache, horizons=horizons)

        for experiment_site, targeted_model in [
            ("layer16", layer16_model),
            ("layer34", layer34_model),
        ]:
            phase = f"{split_name}:{experiment_site}"
            site_rows = _evaluate_block(
                runner=runner,
                cache=cache,
                horizons=horizons,
                split_name=split_name,
                experiment_site=experiment_site,
                broad_model=broad_model,
                targeted_model=targeted_model,
                baseline=baseline,
            )
            for row in site_rows:
                persist(row)

        bad_identity: list[tuple[str, int]] = []
        for row in rows:
            if row["split"] != split_name or row["method"] != "identity":
                continue
            for h in horizons:
                if abs(float(row[f"cont_match_h{h}_uplift_vs_full"]) - float(row[f"no_patch_h{h}"])) > 1e-12:
                    bad_identity.append((row["experiment_site"], h))
                    break
        if bad_identity:
            raise RuntimeError(f"Identity control failed for split={split_name}: {bad_identity[0]}")

        phase = f"{split_name}:complete"
        _flush_artifacts(out_dir, rows, horizons, run_manifest, split_row_counts, phase)

    _flush_artifacts(out_dir, rows, horizons, run_manifest, split_row_counts, phase)
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "num_rows": len(rows),
                "split_row_counts": split_row_counts,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
