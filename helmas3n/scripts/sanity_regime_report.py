#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, Iterable

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from helmas3n.src.gemma.runner import GemmaRunner
from helmas3n.src.gemma.state_extract import load_extraction_config


def _load_prompts(path: Path, prompt_field: str, id_field: str, max_prompts: int) -> list[dict[str, str]]:
    if path.suffix == ".jsonl":
        out = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            out.append({"id": str(item.get(id_field, len(out))), "prompt": str(item.get(prompt_field, item.get("prompt", "")))})
        return out[:max_prompts]

    if path.suffix == ".json":
        raw = json.loads(path.read_text())
        out = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    out.append({"id": str(item.get(id_field, len(out))), "prompt": str(item.get(prompt_field, item.get("prompt", "")))})
                else:
                    out.append({"id": str(len(out)), "prompt": str(item)})
        elif isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list):
            for item in raw["data"]:
                out.append({"id": str(item.get(id_field, len(out))), "prompt": str(item.get(prompt_field, item.get("prompt", "")))})
        return out[:max_prompts]

    out = []
    for i, line in enumerate(path.read_text().splitlines()):
        if line.strip():
            out.append({"id": str(i), "prompt": line.strip()})
    return out[:max_prompts]


def _kl_full_to_low(low_logits: torch.Tensor, full_logits: torch.Tensor) -> float:
    low_logp = F.log_softmax(low_logits.float(), dim=-1)
    full_p = F.softmax(full_logits.float(), dim=-1)
    kl = F.kl_div(low_logp, full_p, reduction="sum")
    return float(kl.item())


def _positions(seq_len: int, last_n: int) -> list[int]:
    start = max(0, seq_len - last_n)
    return list(range(start, seq_len))


def _layer_ids(num_layers: int, stride: int) -> list[int]:
    return list(range(0, num_layers, max(stride, 1)))


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _collect_runtime(
    runner: GemmaRunner,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    regime: str,
    capture_kv: bool,
    capture_logits: bool,
):
    if runner.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runner.device)
        _sync_if_cuda(runner.device)

    t0 = time.perf_counter()
    out = runner.forward_prefix(
        input_ids=input_ids,
        attention_mask=attention_mask,
        regime=regime,
        capture_kv=capture_kv,
        capture_logits=capture_logits,
    )
    _sync_if_cuda(runner.device)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    peak_mem_mb = 0.0
    if runner.device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(runner.device) / (1024 * 1024)

    return out, dt_ms, peak_mem_mb


def run_sanity(
    config_path: Path,
    out_dir: Path,
    max_prompts: int,
    layer_stride: int,
    last_n_positions: int,
    seed: int,
) -> Dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)

    cfg = load_extraction_config(config_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = _load_prompts(
        Path(cfg.prompts_path),
        prompt_field=cfg.prompt_field,
        id_field=cfg.id_field,
        max_prompts=max_prompts,
    )

    runner = GemmaRunner(
        model_name=cfg.model_name,
        device=cfg.device,
        dtype=cfg.dtype,
        trust_remote_code=cfg.trust_remote_code,
        regimes=cfg.regimes,
        model_load_overrides=cfg.model_load_overrides,
        text_only=cfg.text_only,
    )

    num_layers = runner.get_num_layers()
    layers = _layer_ids(num_layers, layer_stride)

    per_layer_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    low_latencies: list[float] = []
    full_latencies: list[float] = []
    low_mem: list[float] = []
    full_mem: list[float] = []

    layer_cos = {layer: [] for layer in layers}
    layer_mse = {layer: [] for layer in layers}
    layer_k_mse = {layer: [] for layer in layers}
    layer_v_mse = {layer: [] for layer in layers}

    for i, item in enumerate(prompts):
        prompt_id = item["id"]
        prompt = item["prompt"]
        if not prompt:
            continue

        input_ids, attention_mask = runner.tokenize(prompt, max_length=cfg.max_length)
        seq_len = int(input_ids.size(1))
        positions = _positions(seq_len, last_n_positions)

        low, low_ms, low_peak = _collect_runtime(
            runner=runner,
            input_ids=input_ids,
            attention_mask=attention_mask,
            regime="low",
            capture_kv=cfg.capture_kv,
            capture_logits=cfg.capture_logits,
        )
        full, full_ms, full_peak = _collect_runtime(
            runner=runner,
            input_ids=input_ids,
            attention_mask=attention_mask,
            regime="full",
            capture_kv=cfg.capture_kv,
            capture_logits=cfg.capture_logits,
        )
        low_latencies.append(low_ms)
        full_latencies.append(full_ms)
        low_mem.append(low_peak)
        full_mem.append(full_peak)

        kl_vals = []
        top1_vals = []
        for pos in positions:
            if low.logits is not None and full.logits is not None:
                low_logits = low.logits[0, pos, :]
                full_logits = full.logits[0, pos, :]
                kl_vals.append(_kl_full_to_low(low_logits, full_logits))
                top1_vals.append(float((low_logits.argmax() == full_logits.argmax()).item()))

        for layer in layers:
            low_bsd = runner.normalize_hidden_state(low.hidden_states[layer + 1])
            full_bsd = runner.normalize_hidden_state(full.hidden_states[layer + 1])
            kv_ready = cfg.capture_kv and low.past_key_values is not None and full.past_key_values is not None
            layer_cos_prompt = []
            layer_mse_prompt = []
            layer_k_mse_prompt = []
            layer_v_mse_prompt = []
            for pos in positions:
                low_h = low_bsd[0, pos, :].float()
                full_h = full_bsd[0, pos, :].float()
                layer_cos_prompt.append(float(F.cosine_similarity(low_h.unsqueeze(0), full_h.unsqueeze(0), dim=-1).item()))
                layer_mse_prompt.append(float(F.mse_loss(low_h, full_h).item()))

                if (
                    kv_ready
                ):
                    low_k, low_v = runner.get_layer_kv_at_position(
                        past_key_values=low.past_key_values,
                        layer_idx=layer,
                        token_position=pos,
                        sequence_length=seq_len,
                    )
                    full_k, full_v = runner.get_layer_kv_at_position(
                        past_key_values=full.past_key_values,
                        layer_idx=layer,
                        token_position=pos,
                        sequence_length=seq_len,
                    )
                    low_k = low_k.float()
                    low_v = low_v.float()
                    full_k = full_k.float()
                    full_v = full_v.float()
                    layer_k_mse_prompt.append(float(F.mse_loss(low_k, full_k).item()))
                    layer_v_mse_prompt.append(float(F.mse_loss(low_v, full_v).item()))

            layer_cos[layer].append(sum(layer_cos_prompt) / max(len(layer_cos_prompt), 1))
            layer_mse[layer].append(sum(layer_mse_prompt) / max(len(layer_mse_prompt), 1))
            if layer_k_mse_prompt:
                layer_k_mse[layer].append(sum(layer_k_mse_prompt) / len(layer_k_mse_prompt))
                layer_v_mse[layer].append(sum(layer_v_mse_prompt) / len(layer_v_mse_prompt))

        prompt_rows.append(
            {
                "prompt_idx": i,
                "prompt_id": prompt_id,
                "seq_len": seq_len,
                "low_latency_ms": low_ms,
                "full_latency_ms": full_ms,
                "low_peak_mem_mb": low_peak,
                "full_peak_mem_mb": full_peak,
                "next_token_kl_full_to_low": sum(kl_vals) / max(len(kl_vals), 1),
                "next_token_top1_agreement": sum(top1_vals) / max(len(top1_vals), 1),
            }
        )

    for layer in layers:
        per_layer_rows.append(
            {
                "layer": layer,
                "residual_cosine_mean": sum(layer_cos[layer]) / max(len(layer_cos[layer]), 1),
                "residual_mse_mean": sum(layer_mse[layer]) / max(len(layer_mse[layer]), 1),
                "k_mse_mean": (sum(layer_k_mse[layer]) / len(layer_k_mse[layer])) if layer_k_mse[layer] else None,
                "v_mse_mean": (sum(layer_v_mse[layer]) / len(layer_v_mse[layer])) if layer_v_mse[layer] else None,
            }
        )

    per_layer_df = pd.DataFrame(per_layer_rows)
    prompt_df = pd.DataFrame(prompt_rows)
    per_layer_df.to_csv(out_dir / "per_layer_metrics.csv", index=False)
    prompt_df.to_csv(out_dir / "per_prompt_metrics.csv", index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(per_layer_df["layer"], per_layer_df["residual_cosine_mean"], marker="o", label="Residual cosine")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Layer")
    plt.ylabel("Cosine")
    plt.title("Low vs Full Residual Cosine by Layer")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "per_layer_residual_cosine.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(per_layer_df["layer"], per_layer_df["residual_mse_mean"], marker="o", color="#b34a3c", label="Residual MSE")
    plt.xlabel("Layer")
    plt.ylabel("MSE")
    plt.title("Low vs Full Residual MSE by Layer")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "per_layer_residual_mse.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    latency_df = pd.DataFrame(
        {
            "regime": ["low", "full"],
            "latency_ms": [sum(low_latencies) / max(len(low_latencies), 1), sum(full_latencies) / max(len(full_latencies), 1)],
            "peak_mem_mb": [sum(low_mem) / max(len(low_mem), 1), sum(full_mem) / max(len(full_mem), 1)],
        }
    )
    plt.bar(latency_df["regime"], latency_df["latency_ms"], color=["#4d88ff", "#e06666"])
    plt.ylabel("Average forward latency (ms)")
    plt.title("Low vs Full Runtime")
    plt.tight_layout()
    plt.savefig(out_dir / "runtime_latency_bar.png", dpi=160)
    plt.close()

    summary = {
        "config": {
            "path": str(config_path),
            "max_prompts": max_prompts,
            "layer_stride": layer_stride,
            "last_n_positions": last_n_positions,
            "seed": seed,
        },
        "regimes": {
            "low": runner.describe_regime("low"),
            "full": runner.describe_regime("full"),
        },
        "global": {
            "num_prompts": int(len(prompt_df)),
            "residual_cosine_mean": float(per_layer_df["residual_cosine_mean"].mean()),
            "residual_mse_mean": float(per_layer_df["residual_mse_mean"].mean()),
            "next_token_kl_full_to_low_mean": float(prompt_df["next_token_kl_full_to_low"].mean()),
            "next_token_top1_agreement_mean": float(prompt_df["next_token_top1_agreement"].mean()),
            "low_latency_ms_mean": float(prompt_df["low_latency_ms"].mean()),
            "full_latency_ms_mean": float(prompt_df["full_latency_ms"].mean()),
            "low_peak_mem_mb_mean": float(prompt_df["low_peak_mem_mb"].mean()),
            "full_peak_mem_mb_mean": float(prompt_df["full_peak_mem_mb"].mean()),
            "latency_delta_full_minus_low_ms": float(prompt_df["full_latency_ms"].mean() - prompt_df["low_latency_ms"].mean()),
            "peak_mem_delta_full_minus_low_mb": float(prompt_df["full_peak_mem_mb"].mean() - prompt_df["low_peak_mem_mb"].mean()),
        },
    }

    (out_dir / "sanity_report.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run low/full regime sanity separation report for RegimeLift")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--max-prompts", type=int, default=50)
    parser.add_argument("--layer-stride", type=int, default=4)
    parser.add_argument("--last-n-positions", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (config_path.parent.parent / "artifacts" / "reports" / "sanity")
    run_sanity(
        config_path=config_path,
        out_dir=out_dir,
        max_prompts=args.max_prompts,
        layer_stride=args.layer_stride,
        last_n_positions=args.last_n_positions,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
