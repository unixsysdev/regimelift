#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from helmas3n.src.data.paired_dataset import PairedStateDataset, paired_collate
from helmas3n.src.eval.decode_resume import (
    resume_with_kv_cache,
    resume_with_residual_patch,
)
from helmas3n.src.eval.handoff_metrics import (
    continuation_match_rate,
    mean_cosine_similarity,
    mean_squared_error,
    sparse_top1_agreement,
)
from helmas3n.src.gemma.runner import GemmaRunner
from helmas3n.src.gemma.state_extract import load_extraction_config
from helmas3n.src.losses.logit_loss import project_topk_logits, sparse_topk_kl_loss
from helmas3n.src.models.uplift_flow import LayerConditionedFlowUplift
from helmas3n.src.models.uplift_baselines import (
    GlobalLinearUplift,
    IdentityUplift,
    LayerwiseMeanDeltaUplift,
)
from helmas3n.src.models.uplift_linear import LayerwiseLinearUplift
from helmas3n.src.models.uplift_lowrank import LayerwiseLowRankUplift
from helmas3n.src.models.uplift_mlp import LayerwiseMLPUplift


def _build_model(kind: str, num_layers: int, hidden_dim: int, model_cfg: Dict[str, Any]) -> torch.nn.Module:
    if kind == "identity":
        return IdentityUplift()
    if kind == "mean_delta":
        return LayerwiseMeanDeltaUplift(num_layers=num_layers, hidden_dim=hidden_dim)
    if kind == "global_linear":
        return GlobalLinearUplift(hidden_dim=hidden_dim, bias=True)
    if kind == "linear":
        return LayerwiseLinearUplift(num_layers=num_layers, hidden_dim=hidden_dim)
    if kind == "lowrank":
        return LayerwiseLowRankUplift(num_layers=num_layers, hidden_dim=hidden_dim, rank=int(model_cfg.get("rank", 64)))
    if kind == "mlp":
        return LayerwiseMLPUplift(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            expansion=float(model_cfg.get("mlp_expansion", 2.0)),
            shared=bool(model_cfg.get("mlp_shared", False)),
        )
    if kind == "flow":
        return LayerConditionedFlowUplift(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            flow_hidden_dim=int(model_cfg.get("flow_hidden_dim", 1024)),
        )
    raise ValueError(f"Unknown model kind: {kind}")


def evaluate(cfg: Dict[str, Any], checkpoint_path: Path, max_batches: int) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PairedStateDataset(cfg["data"]["root_dir"], target=cfg["data"]["target"])
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["data"].get("batch_size", 64)),
        shuffle=False,
        num_workers=0,
        collate_fn=paired_collate,
    )

    sample = dataset[0]
    hidden_dim = int(sample["source_state"].numel())
    num_layers = max(int(dataset.info.num_layers), int(sample["layer"].item()) + 1)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = _build_model(cfg["model"]["kind"], num_layers=num_layers, hidden_dim=hidden_dim, model_cfg=cfg["model"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    logit_head = None
    logit_cfg = cfg.get("logit_head", {})
    if bool(logit_cfg.get("enabled", False)):
        snapshot = Path(logit_cfg.get("snapshot_path", ""))
        if not snapshot.exists():
            raise FileNotFoundError(
                f"logit_head.enabled=true but snapshot_path does not exist: {snapshot}"
            )
        logit_head = torch.load(snapshot, map_location=device)

    mse_vals = []
    cos_vals = []
    top1_low_vs_full_vals = []
    top1_uplift_vs_full_vals = []
    kl_low_vs_full_vals = []
    kl_uplift_vs_full_vals = []
    continuation_vals = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break

            layer = batch["layer"].to(device)
            src = batch["source_state"].to(device).float()
            tgt = batch["target_state"].to(device).float()
            pred = model(src, layer)

            mse_vals.append(mean_squared_error(pred, tgt).item())
            cos_vals.append(mean_cosine_similarity(pred, tgt).item())

            if "full_logits_indices" in batch and "full_logits_values" in batch and "low_logits_indices" in batch and "low_logits_values" in batch:
                top1_low_vs_full_vals.append(
                    sparse_top1_agreement(
                        indices_a=batch["low_logits_indices"],
                        values_a=batch["low_logits_values"],
                        indices_b=batch["full_logits_indices"],
                        values_b=batch["full_logits_values"],
                    ).item()
                )
                if logit_head is not None:
                    full_idx = batch["full_logits_indices"].to(device).long()
                    full_vals = batch["full_logits_values"].to(device).float()
                    pred_vals = project_topk_logits(
                        hidden=pred,
                        token_indices=full_idx,
                        lm_head_weight=logit_head["lm_head_weight"],
                        lm_head_bias=logit_head.get("lm_head_bias"),
                        final_norm_weight=logit_head.get("final_norm_weight"),
                        final_norm_eps=float(logit_head.get("final_norm_eps", 1e-6)),
                    )
                    full_top = full_idx.gather(-1, full_vals.argmax(dim=-1, keepdim=True)).squeeze(-1)
                    uplift_top = full_idx.gather(-1, pred_vals.argmax(dim=-1, keepdim=True)).squeeze(-1)
                    top1_uplift_vs_full_vals.append((uplift_top == full_top).float().mean().item())
                    kl_uplift_vs_full_vals.append(float(sparse_topk_kl_loss(pred_vals, full_vals).item()))

                if "low_on_full_logits_values" in batch:
                    low_on_full_vals = batch["low_on_full_logits_values"].to(device).float()
                    full_vals = batch["full_logits_values"].to(device).float()
                    kl_low_vs_full_vals.append(float(sparse_topk_kl_loss(low_on_full_vals, full_vals).item()))

            if cfg["data"]["target"] == "kv" and "k_full" in batch:
                b = batch["k_full"].shape[0]
                k_full = batch["k_full"].reshape(b, -1)
                pred_k = pred[:, : k_full.size(1)]
                continuation_vals.append((pred_k.argmax(dim=-1) == k_full.argmax(dim=-1)).float().mean().item())

    short_match = float(sum(continuation_vals) / len(continuation_vals)) if continuation_vals else None

    metrics = {
        "state_mse": float(sum(mse_vals) / max(len(mse_vals), 1)),
        "state_cosine": float(sum(cos_vals) / max(len(cos_vals), 1)),
        "next_token_top1_low_vs_full": float(sum(top1_low_vs_full_vals) / max(len(top1_low_vs_full_vals), 1)) if top1_low_vs_full_vals else 0.0,
        "next_token_top1_uplift_vs_full": float(sum(top1_uplift_vs_full_vals) / max(len(top1_uplift_vs_full_vals), 1)) if top1_uplift_vs_full_vals else 0.0,
        "next_token_kl_low_vs_full": float(sum(kl_low_vs_full_vals) / max(len(kl_low_vs_full_vals), 1)) if kl_low_vs_full_vals else None,
        "next_token_kl_uplift_vs_full": float(sum(kl_uplift_vs_full_vals) / max(len(kl_uplift_vs_full_vals), 1)) if kl_uplift_vs_full_vals else None,
        "short_continuation_match_rate": short_match,
        "long_horizon_drift_proxy": (float(1.0 - short_match) if short_match is not None else None),
    }
    return metrics


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


def evaluate_live_handoff(
    train_cfg: Dict[str, Any],
    checkpoint_path: Path,
    extract_cfg_path: Path,
    max_prompts: int,
    max_new_tokens: int,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extract_cfg = load_extraction_config(extract_cfg_path)
    prompts = _load_prompts_from_extract_cfg(extract_cfg_path, max_prompts=max_prompts)

    runner = GemmaRunner(
        model_name=extract_cfg.model_name,
        device=extract_cfg.device,
        dtype=extract_cfg.dtype,
        trust_remote_code=extract_cfg.trust_remote_code,
        regimes=extract_cfg.regimes,
        model_load_overrides=extract_cfg.model_load_overrides,
        text_only=extract_cfg.text_only,
    )

    sample_ds = PairedStateDataset(train_cfg["data"]["root_dir"], target=train_cfg["data"]["target"])
    sample = sample_ds[0]
    hidden_dim = int(sample["source_state"].numel())
    num_layers = max(int(sample_ds.info.num_layers), int(sample["layer"].item()) + 1)
    num_heads = runner.get_num_key_value_heads()
    head_dim = runner.get_head_dim()

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = _build_model(train_cfg["model"]["kind"], num_layers=num_layers, hidden_dim=hidden_dim, model_cfg=train_cfg["model"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    full_match = []
    low_match = []
    next_token_uplift_vs_full = []

    layer = runner.get_num_layers() - 1

    for prompt in prompts:
        input_ids, attention_mask = runner.tokenize(prompt, max_length=extract_cfg.max_length)
        low = runner.forward_prefix(input_ids, attention_mask, regime="low", capture_kv=True, capture_logits=True)
        full = runner.forward_prefix(input_ids, attention_mask, regime="full", capture_kv=True, capture_logits=True)

        if train_cfg["data"]["target"] == "residual":
            low_bsd = runner.normalize_hidden_state(low.hidden_states[layer + 1])
            low_state = low_bsd[0, -1, :].detach().to(device).float().unsqueeze(0)
        else:
            if low.past_key_values is None:
                continue
            last_pos = int(input_ids.size(1) - 1)
            low_k, low_v = runner.get_layer_kv_at_position(
                past_key_values=low.past_key_values,
                layer_idx=layer,
                token_position=last_pos,
                sequence_length=int(input_ids.size(1)),
            )
            lk = low_k.reshape(-1)
            lv = low_v.reshape(-1)
            low_state = torch.cat([lk, lv], dim=0).to(device).float().unsqueeze(0)

        pred = model(low_state, torch.tensor([layer], device=device))

        full_tokens = resume_with_kv_cache(
            runner=runner,
            uplifted_past_key_values=full.past_key_values,
            last_token_ids=input_ids[:, -1:],
            regime="full",
            max_new_tokens=max_new_tokens,
            seed_token_ids=full.logits[:, -1, :].argmax(dim=-1, keepdim=True),
        )
        low_tokens = resume_with_kv_cache(
            runner=runner,
            uplifted_past_key_values=low.past_key_values,
            last_token_ids=input_ids[:, -1:],
            regime="low",
            max_new_tokens=max_new_tokens,
            seed_token_ids=low.logits[:, -1, :].argmax(dim=-1, keepdim=True),
        )

        if train_cfg["data"]["target"] == "residual":
            uplift_tokens = resume_with_residual_patch(
                runner=runner,
                input_ids=input_ids,
                attention_mask=attention_mask,
                patch_map={layer: pred.squeeze(0).detach().cpu()},
                regime="low",
                decode_regime="full",
                max_new_tokens=max_new_tokens,
            )
        else:
            low_past = runner.to_legacy_past_key_values(low.past_key_values)
            if low_past is None:
                continue
            uplift_past = []
            for i, (k, v) in enumerate(low_past):
                k2 = k.clone()
                v2 = v.clone()
                if i == layer and num_heads > 0 and head_dim > 0:
                    pk = pred[:, : num_heads * head_dim].reshape(num_heads, head_dim).to(k2.device, k2.dtype)
                    pv = pred[:, num_heads * head_dim : 2 * num_heads * head_dim].reshape(num_heads, head_dim).to(v2.device, v2.dtype)
                    k2[0, :, -1, :] = pk
                    v2[0, :, -1, :] = pv
                uplift_past.append((k2, v2))
            uplift_tokens = resume_with_kv_cache(
                runner=runner,
                uplifted_past_key_values=tuple(uplift_past),
                last_token_ids=input_ids[:, -1:],
                regime="full",
                max_new_tokens=max_new_tokens,
                seed_token_ids=low.logits[:, -1, :].argmax(dim=-1, keepdim=True),
            )

        full_match.append(continuation_match_rate(full_tokens.cpu(), uplift_tokens.cpu()).item())
        low_match.append(continuation_match_rate(full_tokens.cpu(), low_tokens.cpu()).item())
        if full_tokens.numel() > 0 and uplift_tokens.numel() > 0:
            next_token_uplift_vs_full.append((full_tokens[:, 0] == uplift_tokens[:, 0]).float().mean().item())

    avg_uplift = float(sum(full_match) / max(len(full_match), 1))
    avg_low = float(sum(low_match) / max(len(low_match), 1))
    metrics = {
        "live_prompt_count": len(prompts),
        "short_continuation_match_rate_uplift_vs_full": avg_uplift,
        "short_continuation_match_rate_low_vs_full": avg_low,
        "uplift_delta_over_low": avg_uplift - avg_low,
        "long_horizon_drift_uplift": float(1.0 - avg_uplift),
        "next_token_top1_uplift_vs_full_proxy": float(sum(next_token_uplift_vs_full) / max(len(next_token_uplift_vs_full), 1)),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RegimeLift uplift handoff metrics")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--live-extract-config", type=str, default="")
    parser.add_argument("--live-max-prompts", type=int, default=20)
    parser.add_argument("--live-max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    base_dir = config_path.parent

    data_root = Path(cfg["data"]["root_dir"])
    if not data_root.is_absolute():
        cfg["data"]["root_dir"] = str((base_dir / data_root).resolve())

    out_dir = Path(cfg["experiment"]["out_dir"])
    if not out_dir.is_absolute():
        out_dir = (base_dir / out_dir).resolve()
        cfg["experiment"]["out_dir"] = str(out_dir)

    logit_cfg = cfg.get("logit_head", {})
    if "snapshot_path" in logit_cfg:
        snapshot = Path(logit_cfg["snapshot_path"])
        if not snapshot.is_absolute():
            cfg["logit_head"]["snapshot_path"] = str((base_dir / snapshot).resolve())

    ckpt = Path(args.checkpoint).resolve() if args.checkpoint else Path(cfg["experiment"]["out_dir"]) / "best.pt"

    metrics = evaluate(cfg=cfg, checkpoint_path=ckpt, max_batches=args.max_batches)
    if args.live_extract_config:
        live_metrics = evaluate_live_handoff(
            train_cfg=cfg,
            checkpoint_path=ckpt,
            extract_cfg_path=Path(args.live_extract_config).resolve(),
            max_prompts=args.live_max_prompts,
            max_new_tokens=args.live_max_new_tokens,
        )
        metrics["live_handoff"] = live_metrics
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
