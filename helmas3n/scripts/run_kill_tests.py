#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
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

from helmas3n.scripts.eval_handoff import evaluate
from helmas3n.scripts.sanity_regime_report import run_sanity
from helmas3n.scripts.train_uplift import _build_model, _load_yaml, train
from helmas3n.src.data.paired_dataset import PairedStateDataset
from helmas3n.src.eval.decode_resume import resume_with_kv_cache, resume_with_residual_patch
from helmas3n.src.eval.handoff_metrics import continuation_match_rate
from helmas3n.src.gemma.state_extract import load_extraction_config


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


def _prepare_method_cfg(base_cfg: dict[str, Any], paired_root: Path, out_root: Path, method: str, epochs: int) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["root_dir"] = str(paired_root)
    cfg["experiment"]["name"] = f"killtest_{method}"
    cfg["experiment"]["out_dir"] = str((out_root / method).resolve())
    cfg["model"]["kind"] = method
    logit_snapshot = (paired_root / "logit_head.pt").resolve()
    cfg["logit_head"]["snapshot_path"] = str(logit_snapshot)
    cfg["logit_head"]["enabled"] = logit_snapshot.exists()

    if method in {"identity", "mean_delta"}:
        cfg["training"]["epochs"] = 1
    else:
        cfg["training"]["epochs"] = max(1, int(epochs))

    if method == "lowrank":
        cfg["model"]["rank"] = int(cfg["model"].get("rank", 64))
    if method == "mean_delta":
        cfg["model"]["freeze_mean_delta"] = True
    return cfg


def _load_trained_model(cfg: dict[str, Any], checkpoint: Path) -> tuple[torch.nn.Module, int]:
    dataset = PairedStateDataset(cfg["data"]["root_dir"], target="residual")
    sample = dataset[0]
    hidden_dim = int(sample["source_state"].numel())
    num_layers = max(int(dataset.info.num_layers), int(sample["layer"].item()) + 1)
    model = _build_model(cfg["model"]["kind"], num_layers=num_layers, hidden_dim=hidden_dim, model_cfg=cfg["model"])
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, num_layers - 1


def _evaluate_live_horizons(
    runner,
    prompt_cache: list[dict[str, torch.Tensor]],
    extract_cfg,
    horizons: list[int],
    method_name: str,
    model: torch.nn.Module | None,
    layer: int,
) -> dict[str, Any]:
    max_h = max(horizons)
    device = runner.device
    layer_t = torch.tensor([layer], device=device)

    uplift_sums = {h: 0.0 for h in horizons}
    low2full_sums = {h: 0.0 for h in horizons}
    lowcached_sums = {h: 0.0 for h in horizons}
    lownative_sums = {h: 0.0 for h in horizons}
    next_top1 = []

    total = len(prompt_cache)
    for idx, sample in enumerate(prompt_cache, start=1):
        if idx == 1 or idx % 10 == 0 or idx == total:
            print(f"[killtest] {method_name}: prompt {idx}/{total}")
        input_ids = sample["input_ids"].to(runner.device)
        attention_mask = sample["attention_mask"].to(runner.device)
        full_tokens = sample["full_tokens"]
        low_to_full_tokens = sample["low_to_full_tokens"]
        low_to_full_live_nopatch_tokens = sample["low_to_full_live_nopatch_tokens"]
        low_native_tokens = sample["low_native_tokens"]

        if method_name == "oracle_full_state":
            patch_vec = sample["full_state"]
        else:
            assert model is not None
            with torch.no_grad():
                low_state = sample["low_state"].to(device).float().unsqueeze(0)
                pred = model(low_state, layer_t)
                patch_vec = pred.squeeze(0).detach().cpu()

        _set_seed(int(sample["seed"]))
        uplift_tokens = resume_with_residual_patch(
            runner=runner,
            input_ids=input_ids,
            attention_mask=attention_mask,
            patch_map={layer: patch_vec},
            regime="low",
            decode_regime="full",
            max_new_tokens=max_h,
        ).cpu()

        for h in horizons:
            uplift_sums[h] += float(continuation_match_rate(full_tokens[:, :h], uplift_tokens[:, :h]).item())
            low2full_sums[h] += float(
                continuation_match_rate(full_tokens[:, :h], low_to_full_live_nopatch_tokens[:, :h]).item()
            )
            lowcached_sums[h] += float(continuation_match_rate(full_tokens[:, :h], low_to_full_tokens[:, :h]).item())
            lownative_sums[h] += float(continuation_match_rate(full_tokens[:, :h], low_native_tokens[:, :h]).item())

        if full_tokens.numel() > 0 and uplift_tokens.numel() > 0:
            next_top1.append(float((full_tokens[:, 0] == uplift_tokens[:, 0]).float().mean().item()))

    denom = max(len(prompt_cache), 1)
    out = {
        "method": method_name,
        "prompt_count": len(prompt_cache),
        "next_token_top1_uplift_vs_full_live": float(sum(next_top1) / max(len(next_top1), 1)) if next_top1 else 0.0,
    }
    for h in horizons:
        out[f"cont_match_h{h}_uplift_vs_full"] = uplift_sums[h] / denom
        out[f"cont_match_h{h}_low2full_vs_full"] = low2full_sums[h] / denom
        out[f"cont_match_h{h}_lowcached_vs_full"] = lowcached_sums[h] / denom
        out[f"cont_match_h{h}_lownative_vs_full"] = lownative_sums[h] / denom
        out[f"delta_h{h}"] = out[f"cont_match_h{h}_uplift_vs_full"] - out[f"cont_match_h{h}_low2full_vs_full"]
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _build_prompt_cache(
    runner,
    prompts: list[str],
    max_length: int,
    layer: int,
    max_h: int,
    base_seed: int,
) -> list[dict[str, torch.Tensor]]:
    cache = []
    total = len(prompts)
    for idx, prompt in enumerate(prompts, start=1):
        if idx == 1 or idx % 10 == 0 or idx == total:
            print(f"[killtest] building prompt cache {idx}/{total}")
        prompt_seed = int(base_seed + idx * 1009)
        _set_seed(prompt_seed)
        input_ids, attention_mask = runner.tokenize(prompt, max_length=max_length)
        low = runner.forward_prefix(input_ids, attention_mask, regime="low", capture_kv=True, capture_logits=True)
        _set_seed(prompt_seed + 1)
        full = runner.forward_prefix(input_ids, attention_mask, regime="full", capture_kv=True, capture_logits=True)

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
        _set_seed(prompt_seed)
        low_to_full_live_nopatch_tokens = resume_with_residual_patch(
            runner=runner,
            input_ids=input_ids,
            attention_mask=attention_mask,
            patch_map={},
            regime="low",
            decode_regime="full",
            max_new_tokens=max_h,
        ).cpu()

        cache.append(
            {
                "input_ids": input_ids.cpu(),
                "attention_mask": attention_mask.cpu(),
                "low_state": runner.normalize_hidden_state(low.hidden_states[layer + 1])[0, -1, :].detach().cpu(),
                "full_state": runner.normalize_hidden_state(full.hidden_states[layer + 1])[0, -1, :].detach().cpu(),
                "low_native_tokens": low_native_tokens,
                "low_to_full_tokens": low_to_full_tokens,
                "low_to_full_live_nopatch_tokens": low_to_full_live_nopatch_tokens,
                "full_tokens": full_tokens,
                "seed": prompt_seed,
            }
        )
    return cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HeLMAS-3n kill-test table with oracle row and horizon metrics")
    parser.add_argument("--extract-config", type=str, default="configs/extract_killtest.yaml")
    parser.add_argument("--train-template", type=str, default="configs/train_residual.yaml")
    parser.add_argument("--out-dir", type=str, default="artifacts/reports/killtest")
    parser.add_argument("--max-prompts", type=int, default=80)
    parser.add_argument("--live-max-prompts", type=int, default=None)
    parser.add_argument("--sanity-max-prompts", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--offline-max-batches", type=int, default=100)
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    extract_cfg_path = Path(args.extract_config)
    if not extract_cfg_path.is_absolute():
        extract_cfg_path = (script_dir.parent / args.extract_config).resolve()
    train_template_path = Path(args.train_template)
    if not train_template_path.is_absolute():
        train_template_path = (script_dir.parent / args.train_template).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (script_dir.parent / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    extract_cfg = load_extraction_config(extract_cfg_path)
    paired_root = Path(extract_cfg.out_dir).resolve()

    base_train_cfg = _load_yaml(train_template_path)
    ckpt_root = out_dir / "checkpoints"
    methods = ["identity", "mean_delta", "global_linear", "linear", "lowrank", "mlp"]
    horizons = [1, 4, 8, 16]
    live_max_prompts = args.live_max_prompts if args.live_max_prompts is not None else args.max_prompts

    # Ensure checkpoints exist for all required methods.
    method_cfgs: dict[str, dict[str, Any]] = {}
    method_ckpts: dict[str, Path] = {}
    for method in methods:
        cfg = _prepare_method_cfg(base_train_cfg, paired_root=paired_root, out_root=ckpt_root, method=method, epochs=args.epochs)
        out_ckpt = Path(cfg["experiment"]["out_dir"]) / "last.pt"
        if args.retrain or not out_ckpt.exists():
            print(f"[killtest] training {method} -> {out_ckpt}")
            train(cfg)
        method_cfgs[method] = cfg
        method_ckpts[method] = out_ckpt

    # Offline alignment metrics from paired dataset.
    rows: list[dict[str, Any]] = []
    for method in methods:
        print(f"[killtest] offline eval {method}")
        off = evaluate(cfg=method_cfgs[method], checkpoint_path=method_ckpts[method], max_batches=args.offline_max_batches)
        row = {
            "method": method,
            "state_cosine": off["state_cosine"],
            "state_mse": off["state_mse"],
            "next_token_kl_uplift_vs_full": off["next_token_kl_uplift_vs_full"],
            "next_token_top1_uplift_vs_full_offline": off["next_token_top1_uplift_vs_full"],
        }
        rows.append(row)

    # Live behavioral metrics with a single shared runner.
    from helmas3n.src.gemma.runner import GemmaRunner

    runner = GemmaRunner(
        model_name=extract_cfg.model_name,
        device=extract_cfg.device,
        dtype=extract_cfg.dtype,
        trust_remote_code=extract_cfg.trust_remote_code,
        regimes=extract_cfg.regimes,
        model_load_overrides=extract_cfg.model_load_overrides,
        text_only=extract_cfg.text_only,
    )
    prompts = _load_prompts_from_extract_cfg(extract_cfg_path, max_prompts=live_max_prompts)
    layer = runner.get_num_layers() - 1
    prompt_cache = _build_prompt_cache(
        runner=runner,
        prompts=prompts,
        max_length=extract_cfg.max_length,
        layer=layer,
        max_h=max(horizons),
        base_seed=extract_cfg.seed,
    )

    for row in rows:
        method = row["method"]
        print(f"[killtest] live eval {method}")
        model, _ = _load_trained_model(method_cfgs[method], method_ckpts[method])
        model.to(runner.device)
        live = _evaluate_live_horizons(
            runner=runner,
            prompt_cache=prompt_cache,
            extract_cfg=extract_cfg,
            horizons=horizons,
            method_name=method,
            model=model,
            layer=layer,
        )
        row.update(live)

    oracle_live = _evaluate_live_horizons(
        runner=runner,
        prompt_cache=prompt_cache,
        extract_cfg=extract_cfg,
        horizons=horizons,
        method_name="oracle_full_state",
        model=None,
        layer=layer,
    )
    rows.append(
        {
            "method": "oracle_full_state",
            "state_cosine": None,
            "state_mse": None,
            "next_token_kl_uplift_vs_full": None,
            "next_token_top1_uplift_vs_full_offline": None,
            **oracle_live,
        }
    )

    # Add low/full native rows from sanity + live baselines.
    sanity = run_sanity(
        config_path=extract_cfg_path,
        out_dir=out_dir / "sanity",
        max_prompts=min(args.sanity_max_prompts, args.max_prompts),
        layer_stride=extract_cfg.layer_stride,
        last_n_positions=min(extract_cfg.last_n_positions, 16),
        seed=extract_cfg.seed,
    )
    low_row = {
        "method": "low_native",
        "state_cosine": sanity["global"]["residual_cosine_mean"],
        "state_mse": sanity["global"]["residual_mse_mean"],
        "next_token_kl_uplift_vs_full": sanity["global"]["next_token_kl_full_to_low_mean"],
        "next_token_top1_uplift_vs_full_offline": sanity["global"]["next_token_top1_agreement_mean"],
    }
    full_row = {
        "method": "full_native",
        "state_cosine": 1.0,
        "state_mse": 0.0,
        "next_token_kl_uplift_vs_full": 0.0,
        "next_token_top1_uplift_vs_full_offline": 1.0,
    }
    low2full_row = {
        "method": "low_to_full_no_patch",
        "state_cosine": None,
        "state_mse": None,
        "next_token_kl_uplift_vs_full": None,
        "next_token_top1_uplift_vs_full_offline": None,
    }
    # Reuse low/full continuation from oracle eval payload.
    for h in horizons:
        low_row[f"cont_match_h{h}_uplift_vs_full"] = oracle_live[f"cont_match_h{h}_lownative_vs_full"]
        low_row[f"cont_match_h{h}_low2full_vs_full"] = oracle_live[f"cont_match_h{h}_low2full_vs_full"]
        low_row[f"cont_match_h{h}_lowcached_vs_full"] = oracle_live[f"cont_match_h{h}_lowcached_vs_full"]
        low_row[f"cont_match_h{h}_lownative_vs_full"] = oracle_live[f"cont_match_h{h}_lownative_vs_full"]
        low_row[f"delta_h{h}"] = low_row[f"cont_match_h{h}_uplift_vs_full"] - low_row[f"cont_match_h{h}_low2full_vs_full"]
        low2full_row[f"cont_match_h{h}_uplift_vs_full"] = oracle_live[f"cont_match_h{h}_low2full_vs_full"]
        low2full_row[f"cont_match_h{h}_low2full_vs_full"] = oracle_live[f"cont_match_h{h}_low2full_vs_full"]
        low2full_row[f"cont_match_h{h}_lowcached_vs_full"] = oracle_live[f"cont_match_h{h}_lowcached_vs_full"]
        low2full_row[f"cont_match_h{h}_lownative_vs_full"] = oracle_live[f"cont_match_h{h}_lownative_vs_full"]
        low2full_row[f"delta_h{h}"] = 0.0
        full_row[f"cont_match_h{h}_uplift_vs_full"] = 1.0
        full_row[f"cont_match_h{h}_low2full_vs_full"] = oracle_live[f"cont_match_h{h}_low2full_vs_full"]
        full_row[f"cont_match_h{h}_lowcached_vs_full"] = oracle_live[f"cont_match_h{h}_lowcached_vs_full"]
        full_row[f"cont_match_h{h}_lownative_vs_full"] = oracle_live[f"cont_match_h{h}_lownative_vs_full"]
        full_row[f"delta_h{h}"] = 1.0 - oracle_live[f"cont_match_h{h}_low2full_vs_full"]
    low_row["prompt_count"] = len(prompts)
    low2full_row["prompt_count"] = len(prompts)
    full_row["prompt_count"] = len(prompts)
    low_row["next_token_top1_uplift_vs_full_live"] = oracle_live["cont_match_h1_lownative_vs_full"]
    low2full_row["next_token_top1_uplift_vs_full_live"] = oracle_live["cont_match_h1_low2full_vs_full"]
    full_row["next_token_top1_uplift_vs_full_live"] = 1.0
    rows = [low_row, low2full_row] + rows + [full_row]

    curves = []
    for row in rows:
        for h in horizons:
            curves.append(
                {
                    "method": row["method"],
                    "horizon": h,
                    "match_uplift_vs_full": row.get(f"cont_match_h{h}_uplift_vs_full"),
                    "match_low2full_vs_full": row.get(f"cont_match_h{h}_low2full_vs_full"),
                    "match_lowcached_vs_full": row.get(f"cont_match_h{h}_lowcached_vs_full"),
                    "match_lownative_vs_full": row.get(f"cont_match_h{h}_lownative_vs_full"),
                    "delta_over_low": row.get(f"delta_h{h}"),
                }
            )

    (out_dir / "kill_table.json").write_text(json.dumps(rows, indent=2))
    _write_csv(out_dir / "kill_table.csv", rows)
    _write_csv(out_dir / "horizon_curves.csv", curves)
    print(json.dumps({"out_dir": str(out_dir), "rows": len(rows), "prompts": len(prompts)}, indent=2))


if __name__ == "__main__":
    main()
