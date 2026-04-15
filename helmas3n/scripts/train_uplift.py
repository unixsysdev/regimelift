#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from helmas3n.src.data.paired_dataset import PairedStateDataset, paired_collate
from helmas3n.src.eval.handoff_metrics import mean_cosine_similarity, mean_squared_error
from helmas3n.src.losses.attention_loss import (
    attention_output_consistency_loss,
    direct_kv_consistency_loss,
)
from helmas3n.src.losses.logit_loss import project_topk_logits, sparse_topk_kl_loss
from helmas3n.src.losses.state_loss import state_reconstruction_loss
from helmas3n.src.models.uplift_flow import LayerConditionedFlowUplift
from helmas3n.src.models.uplift_baselines import (
    GlobalLinearUplift,
    IdentityUplift,
    LayerwiseMeanDeltaUplift,
)
from helmas3n.src.models.uplift_linear import LayerwiseLinearUplift
from helmas3n.src.models.uplift_lowrank import LayerwiseLowRankUplift
from helmas3n.src.models.uplift_mlp import LayerwiseMLPUplift


@dataclass
class TrainContext:
    cfg: Dict[str, Any]
    device: torch.device
    model: nn.Module
    optimizer: torch.optim.Optimizer | None
    train_loader: DataLoader
    val_loader: DataLoader
    dataset_info: Dict[str, Any]
    logit_head: Dict[str, torch.Tensor] | None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    base_dir = path.parent

    data_root = Path(cfg["data"]["root_dir"])
    if not data_root.is_absolute():
        cfg["data"]["root_dir"] = str((base_dir / data_root).resolve())

    out_dir = Path(cfg["experiment"]["out_dir"])
    if not out_dir.is_absolute():
        cfg["experiment"]["out_dir"] = str((base_dir / out_dir).resolve())

    logit_cfg = cfg.get("logit_head", {})
    if "snapshot_path" in logit_cfg:
        snapshot = Path(logit_cfg["snapshot_path"])
        if not snapshot.is_absolute():
            cfg["logit_head"]["snapshot_path"] = str((base_dir / snapshot).resolve())

    return cfg


def _load_logit_head(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    out: Dict[str, torch.Tensor] = {}
    for k in ["lm_head_weight", "lm_head_bias", "final_norm_weight", "final_norm_bias"]:
        if payload.get(k) is not None:
            out[k] = payload[k].to(device)
    out["final_norm_eps"] = torch.tensor(float(payload.get("final_norm_eps", 1e-6)), device=device)
    return out


def _build_model(kind: str, num_layers: int, hidden_dim: int, model_cfg: Dict[str, Any]) -> nn.Module:
    if kind == "identity":
        return IdentityUplift()
    if kind == "mean_delta":
        return LayerwiseMeanDeltaUplift(num_layers=num_layers, hidden_dim=hidden_dim)
    if kind == "global_linear":
        return GlobalLinearUplift(hidden_dim=hidden_dim, bias=True)
    if kind == "linear":
        return LayerwiseLinearUplift(num_layers=num_layers, hidden_dim=hidden_dim)
    if kind == "lowrank":
        return LayerwiseLowRankUplift(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            rank=int(model_cfg.get("rank", 64)),
        )
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


def _initialize_mean_delta(
    model: LayerwiseMeanDeltaUplift,
    dataset,
    num_layers: int,
    hidden_dim: int,
    device: torch.device,
) -> None:
    sums = torch.zeros(num_layers, hidden_dim, dtype=torch.float32)
    counts = torch.zeros(num_layers, dtype=torch.float32)

    indices = getattr(dataset, "indices", None)
    if indices is None:
        indices = range(len(dataset))

    base_dataset = getattr(dataset, "dataset", dataset)
    for idx in indices:
        sample = base_dataset[idx]
        layer = int(sample["layer"].item())
        delta = (sample["target_state"] - sample["source_state"]).float()
        sums[layer] += delta
        counts[layer] += 1.0

    counts = counts.clamp_min(1.0).unsqueeze(-1)
    mean_delta = (sums / counts).to(device)
    model.set_delta(mean_delta)


def _prepare_context(cfg: Dict[str, Any]) -> TrainContext:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _set_seed(int(cfg.get("seed", 7)))

    dataset = PairedStateDataset(cfg["data"]["root_dir"], target=cfg["data"]["target"])
    sample = dataset[0]
    hidden_dim = int(sample["source_state"].numel())
    num_layers = max(int(dataset.info.num_layers), int(sample["layer"].item()) + 1)

    n_total = len(dataset)
    n_val = max(1, int(n_total * 0.1))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["data"].get("batch_size", 64)),
        shuffle=True,
        num_workers=int(cfg["data"].get("num_workers", 0)),
        collate_fn=paired_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["data"].get("batch_size", 64)),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 0)),
        collate_fn=paired_collate,
    )

    model = _build_model(
        kind=cfg["model"]["kind"],
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        model_cfg=cfg["model"],
    ).to(device)

    if cfg["data"]["target"] == "kv" and cfg["model"]["kind"] == "linear" and hidden_dim >= 1024:
        raise ValueError(
            "KV + linear uplift is prohibitively large at this hidden_dim. "
            "Use model.kind=lowrank or model.kind=mlp."
        )

    if cfg["model"]["kind"] == "mean_delta":
        _initialize_mean_delta(
            model=model,  # type: ignore[arg-type]
            dataset=train_ds,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            device=device,
        )
        if bool(cfg["model"].get("freeze_mean_delta", True)):
            for p in model.parameters():
                p.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = None
    if trainable_params:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(cfg["optimizer"].get("lr", 3e-4)),
            weight_decay=float(cfg["optimizer"].get("weight_decay", 0.0)),
        )

    logit_head = None
    logit_cfg = cfg.get("logit_head", {})
    if bool(logit_cfg.get("enabled", False)):
        path = Path(logit_cfg["snapshot_path"])
        if not path.exists():
            raise FileNotFoundError(
                f"logit_head.enabled=true but snapshot_path does not exist: {path}"
            )
        logit_head = _load_logit_head(path, device)

    dataset_info = {
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_samples": n_total,
        "target": cfg["data"]["target"],
    }

    return TrainContext(
        cfg=cfg,
        device=device,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        dataset_info=dataset_info,
        logit_head=logit_head,
    )


def _compute_batch_loss(ctx: TrainContext, batch: Dict[str, Any]) -> tuple[torch.Tensor, Dict[str, float]]:
    cfg = ctx.cfg
    loss_cfg = cfg.get("loss", {})

    layer = batch["layer"].to(ctx.device)
    source_state = batch["source_state"].to(ctx.device).float()
    target_state = batch["target_state"].to(ctx.device).float()

    pred = ctx.model(source_state, layer)
    state_losses = state_reconstruction_loss(
        pred=pred,
        target=target_state,
        mse_weight=float(loss_cfg.get("mse_weight", 1.0)),
        cosine_weight=float(loss_cfg.get("cosine_weight", 0.0)),
    )
    total = state_losses["total"]

    logs = {
        "loss_total": float(total.item()),
        "loss_state_mse": float(state_losses["mse"].item()),
        "loss_state_cos": float(state_losses["cosine"].item()),
    }

    logit_weight = float(loss_cfg.get("logit_weight", 0.0))
    if (
        logit_weight > 0
        and ctx.logit_head is not None
        and "full_logits_indices" in batch
        and "full_logits_values" in batch
    ):
        mask = torch.ones(layer.size(0), dtype=torch.bool, device=ctx.device)
        if bool(loss_cfg.get("logit_only_final_layer", True)):
            mask = layer == (ctx.dataset_info["num_layers"] - 1)

        if mask.any():
            pred_hidden = pred[mask]
            indices = batch["full_logits_indices"].to(ctx.device)[mask].long()
            tgt_vals = batch["full_logits_values"].to(ctx.device)[mask].float()

            pred_vals = project_topk_logits(
                hidden=pred_hidden,
                token_indices=indices,
                lm_head_weight=ctx.logit_head["lm_head_weight"],
                lm_head_bias=ctx.logit_head.get("lm_head_bias"),
                final_norm_weight=ctx.logit_head.get("final_norm_weight"),
                final_norm_eps=float(ctx.logit_head["final_norm_eps"].item()),
            )
            logit_kl = sparse_topk_kl_loss(pred_vals, tgt_vals)
            total = total + logit_weight * logit_kl
            logs["loss_logit_kl"] = float(logit_kl.item())

    attn_weight = float(loss_cfg.get("attention_weight", 0.0))
    if (
        attn_weight > 0
        and cfg["data"]["target"] == "kv"
        and "k_low" in batch
        and "k_full" in batch
        and "v_full" in batch
    ):
        k_low = batch["k_low"].to(ctx.device).float()
        k_full = batch["k_full"].to(ctx.device).float()
        v_full = batch["v_full"].to(ctx.device).float()

        b, h, d = k_low.shape
        pred_k, pred_v = torch.split(pred.reshape(b, 2, h, d), 1, dim=1)
        pred_k = pred_k.squeeze(1)
        pred_v = pred_v.squeeze(1)

        prompt_ids = batch["prompt_id"]
        layer_ids = batch["layer"].tolist()
        pos_ids = batch["token_position"].tolist()

        grouped_indices: dict[tuple[str, int], list[tuple[int, int]]] = {}
        for idx, (pid, lid, pos) in enumerate(zip(prompt_ids, layer_ids, pos_ids)):
            grouped_indices.setdefault((str(pid), int(lid)), []).append((int(pos), idx))

        attn_group_losses = []
        used = torch.zeros(b, dtype=torch.bool, device=ctx.device)
        for _, members in grouped_indices.items():
            if len(members) < 2:
                continue
            members = sorted(members, key=lambda t: t[0])
            idxs = torch.tensor([m[1] for m in members], dtype=torch.long, device=ctx.device)
            used[idxs] = True
            g = idxs.numel()

            q = k_low[idxs]  # [G, H, D]
            pk = pred_k[idxs]  # [G, H, D]
            pv = pred_v[idxs]
            tk = k_full[idxs]
            tv = v_full[idxs]

            # Build sequence keys/values per group so attention is over S=G (non-degenerate).
            pk_seq = pk.permute(1, 0, 2).unsqueeze(0).expand(g, -1, -1, -1)  # [G, H, S, D]
            pv_seq = pv.permute(1, 0, 2).unsqueeze(0).expand(g, -1, -1, -1)
            tk_seq = tk.permute(1, 0, 2).unsqueeze(0).expand(g, -1, -1, -1)
            tv_seq = tv.permute(1, 0, 2).unsqueeze(0).expand(g, -1, -1, -1)

            attn_losses = attention_output_consistency_loss(
                query=q,
                pred_k=pk_seq,
                pred_v=pv_seq,
                target_k=tk_seq,
                target_v=tv_seq,
                mse_weight=1.0,
                cosine_weight=0.0,
            )
            attn_group_losses.append(attn_losses["total"])

        fallback_mask = ~used
        direct_losses = direct_kv_consistency_loss(
            pred_k=pred_k[fallback_mask] if fallback_mask.any() else pred_k[:0],
            pred_v=pred_v[fallback_mask] if fallback_mask.any() else pred_v[:0],
            target_k=k_full[fallback_mask] if fallback_mask.any() else k_full[:0],
            target_v=v_full[fallback_mask] if fallback_mask.any() else v_full[:0],
            mse_weight=1.0,
            cosine_weight=0.0,
        ) if fallback_mask.any() else None

        attn_total = torch.tensor(0.0, device=ctx.device)
        if attn_group_losses:
            attn_total = attn_total + torch.stack(attn_group_losses).mean()
        if direct_losses is not None:
            attn_total = attn_total + direct_losses["total"]
            logs["loss_kv_direct_mse"] = float(direct_losses["mse"].item())
        total = total + attn_weight * attn_total
        logs["loss_attention"] = float(attn_total.item())

    logs["loss_total"] = float(total.item())
    return total, logs


def _evaluate(ctx: TrainContext) -> Dict[str, float]:
    ctx.model.eval()
    mse_values = []
    cos_values = []
    max_batches = int(ctx.cfg.get("evaluation", {}).get("max_eval_batches", 50))

    with torch.no_grad():
        for i, batch in enumerate(ctx.val_loader):
            if i >= max_batches:
                break
            layer = batch["layer"].to(ctx.device)
            src = batch["source_state"].to(ctx.device).float()
            tgt = batch["target_state"].to(ctx.device).float()
            pred = ctx.model(src, layer)
            mse_values.append(mean_squared_error(pred, tgt).item())
            cos_values.append(mean_cosine_similarity(pred, tgt).item())

    return {
        "val_mse": float(sum(mse_values) / max(len(mse_values), 1)),
        "val_cosine": float(sum(cos_values) / max(len(cos_values), 1)),
    }


def train(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ctx = _prepare_context(cfg)
    out_dir = Path(cfg["experiment"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = math.inf
    global_step = 0
    history = []

    epochs = int(cfg["training"].get("epochs", 5))
    eval_every = int(cfg["training"].get("eval_every_steps", 200))
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))

    for epoch in range(epochs):
        ctx.model.train()
        for batch in ctx.train_loader:
            if ctx.optimizer is not None:
                ctx.optimizer.zero_grad(set_to_none=True)
            loss, logs = _compute_batch_loss(ctx, batch)
            if ctx.optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ctx.model.parameters(), grad_clip)
                ctx.optimizer.step()

            global_step += 1
            if global_step % eval_every == 0:
                metrics = _evaluate(ctx)
                row = {"step": global_step, "epoch": epoch, **logs, **metrics}
                history.append(row)
                print(json.dumps(row))

                if metrics["val_mse"] < best_val:
                    best_val = metrics["val_mse"]
                    torch.save(
                        {
                            "model_state": ctx.model.state_dict(),
                            "config": cfg,
                            "dataset_info": ctx.dataset_info,
                            "step": global_step,
                            "epoch": epoch,
                            "best_val_mse": best_val,
                        },
                        out_dir / "best.pt",
                    )

    final_metrics = _evaluate(ctx)
    summary = {
        "best_val_mse": best_val,
        "final_val_mse": final_metrics["val_mse"],
        "final_val_cosine": final_metrics["val_cosine"],
        "history": history,
        "dataset_info": ctx.dataset_info,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    torch.save(
        {
            "model_state": ctx.model.state_dict(),
            "config": cfg,
            "dataset_info": ctx.dataset_info,
            "final_metrics": final_metrics,
        },
        out_dir / "last.pt",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HeLMAS-3n uplift baseline")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    summary = train(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
