from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
import yaml

from helmas3n.src.gemma.runner import GemmaRunner


@dataclass
class ExtractionConfig:
    seed: int
    model_name: str
    device: str
    dtype: str
    trust_remote_code: bool
    model_load_overrides: dict[str, Any]
    text_only: bool
    regimes: dict[str, dict[str, Any]]
    prompts_path: str
    prompt_field: str
    id_field: str
    max_prompts: int
    max_length: int
    layer_stride: int
    include_layers: list[int]
    last_n_positions: int
    include_positions: list[int]
    capture_kv: bool
    capture_logits: bool
    logits_topk: int
    out_dir: str
    shard_size: int
    save_logit_head_snapshot: bool


class ShardWriter:
    def __init__(self, out_dir: Path, shard_size: int) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.buffer: Dict[str, list[Any]] = {}
        self.shards: list[dict[str, Any]] = []
        self.total_samples = 0
        self._shard_idx = 0

    def add(self, sample: Dict[str, Any]) -> None:
        for key, value in sample.items():
            self.buffer.setdefault(key, []).append(value)

        if len(next(iter(self.buffer.values()))) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return

        num_samples = len(next(iter(self.buffer.values())))
        shard_obj: Dict[str, Any] = {}
        for key, values in self.buffer.items():
            first = values[0]
            if torch.is_tensor(first):
                shard_obj[key] = torch.stack(values)
            elif isinstance(first, int):
                shard_obj[key] = torch.tensor(values, dtype=torch.long)
            elif isinstance(first, float):
                shard_obj[key] = torch.tensor(values, dtype=torch.float32)
            else:
                shard_obj[key] = values

        shard_name = f"shard_{self._shard_idx:05d}.pt"
        torch.save(shard_obj, self.out_dir / shard_name)
        self.shards.append({"path": shard_name, "num_samples": num_samples})
        self.total_samples += num_samples
        self._shard_idx += 1
        self.buffer = {}

    def finalize(self, extra_manifest: Dict[str, Any]) -> Dict[str, Any]:
        self.flush()
        manifest = {
            "format": "helmas3n.paired.v1",
            "num_samples": self.total_samples,
            "shards": self.shards,
            **extra_manifest,
        }
        (self.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest


def load_extraction_config(path: str | Path) -> ExtractionConfig:
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    base_dir = path.parent

    prompts_path = Path(cfg["data"]["prompts_path"])
    if not prompts_path.is_absolute():
        prompts_path = (base_dir / prompts_path).resolve()

    out_dir = Path(cfg["output"]["dir"])
    if not out_dir.is_absolute():
        out_dir = (base_dir / out_dir).resolve()

    return ExtractionConfig(
        seed=int(cfg.get("seed", 7)),
        model_name=cfg["model"]["name"],
        device=cfg["model"].get("device", "cuda"),
        dtype=cfg["model"].get("dtype", "bfloat16"),
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", True)),
        model_load_overrides=dict(cfg["model"].get("load_overrides", {})),
        text_only=bool(cfg["model"].get("text_only", True)),
        regimes=cfg.get("activation", {}),
        prompts_path=str(prompts_path),
        prompt_field=cfg["data"].get("prompt_field", "prompt"),
        id_field=cfg["data"].get("id_field", "id"),
        max_prompts=int(cfg["data"].get("max_prompts", 200)),
        max_length=int(cfg["data"].get("max_length", 1024)),
        layer_stride=int(cfg["sampling"].get("layer_stride", 4)),
        include_layers=list(cfg["sampling"].get("include_layers", [])),
        last_n_positions=int(cfg["sampling"].get("last_n_positions", 32)),
        include_positions=list(cfg["sampling"].get("include_positions", [])),
        capture_kv=bool(cfg["targets"].get("capture_kv", True)),
        capture_logits=bool(cfg["targets"].get("capture_logits", True)),
        logits_topk=int(cfg["targets"].get("logits_topk", 128)),
        out_dir=str(out_dir),
        shard_size=int(cfg["output"].get("shard_size", 4096)),
        save_logit_head_snapshot=bool(cfg["output"].get("save_logit_head_snapshot", True)),
    )


def _load_prompts(path: Path, prompt_field: str, id_field: str, max_prompts: int) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)

    data: list[dict[str, str]] = []
    if path.suffix == ".jsonl":
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            text = str(item.get(prompt_field, item.get("prompt", "")))
            pid = str(item.get(id_field, len(data)))
            data.append({"id": pid, "prompt": text})
    elif path.suffix == ".json":
        raw = json.loads(path.read_text())
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    text = str(item.get(prompt_field, item.get("prompt", "")))
                    pid = str(item.get(id_field, len(data)))
                else:
                    text = str(item)
                    pid = str(len(data))
                data.append({"id": pid, "prompt": text})
        elif isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list):
            for item in raw["data"]:
                text = str(item.get(prompt_field, item.get("prompt", "")))
                pid = str(item.get(id_field, len(data)))
                data.append({"id": pid, "prompt": text})
        else:
            raise ValueError(f"Unsupported JSON format in {path}")
    else:
        for i, line in enumerate(path.read_text().splitlines()):
            if line.strip():
                data.append({"id": str(i), "prompt": line.strip()})

    return data[:max_prompts]


def _sample_layers(num_layers: int, stride: int, include_layers: Iterable[int]) -> list[int]:
    selected = set(range(0, num_layers, max(stride, 1)))
    selected.update(int(x) for x in include_layers if 0 <= int(x) < num_layers)
    return sorted(selected)


def _sample_positions(seq_len: int, last_n: int, include_positions: Iterable[int]) -> list[int]:
    start = max(0, seq_len - max(last_n, 1))
    selected = set(range(start, seq_len))
    selected.update(int(x) for x in include_positions if 0 <= int(x) < seq_len)
    return sorted(selected)


def _topk_logits(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    values, indices = torch.topk(logits, k=min(k, logits.numel()), dim=-1)
    return values.detach().cpu().float(), indices.detach().cpu().long()


def collect_paired_states(config: ExtractionConfig) -> dict[str, Any]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    runner = GemmaRunner(
        model_name=config.model_name,
        device=config.device,
        dtype=config.dtype,
        trust_remote_code=config.trust_remote_code,
        regimes=config.regimes,
        model_load_overrides=config.model_load_overrides,
        text_only=config.text_only,
    )

    out_dir = Path(config.out_dir)
    writer = ShardWriter(out_dir=out_dir, shard_size=config.shard_size)

    if config.save_logit_head_snapshot:
        runner.export_logit_head_snapshot(out_dir / "logit_head.pt")

    prompts = _load_prompts(
        Path(config.prompts_path),
        prompt_field=config.prompt_field,
        id_field=config.id_field,
        max_prompts=config.max_prompts,
    )

    first_kv_shape = None
    num_layers = runner.get_num_layers()
    hidden_dim = runner.get_hidden_dim()

    for item in prompts:
        prompt_id = item["id"]
        prompt = item["prompt"]

        input_ids, attention_mask = runner.tokenize(prompt, max_length=config.max_length)
        seq_len = input_ids.size(1)

        low = runner.forward_prefix(
            input_ids=input_ids,
            attention_mask=attention_mask,
            regime="low",
            capture_kv=config.capture_kv,
            capture_logits=config.capture_logits,
        )
        full = runner.forward_prefix(
            input_ids=input_ids,
            attention_mask=attention_mask,
            regime="full",
            capture_kv=config.capture_kv,
            capture_logits=config.capture_logits,
        )

        layers = _sample_layers(num_layers=num_layers, stride=config.layer_stride, include_layers=config.include_layers)
        positions = _sample_positions(seq_len=seq_len, last_n=config.last_n_positions, include_positions=config.include_positions)

        low_logits_topk = {}
        full_logits_topk = {}
        if config.capture_logits and low.logits is not None and full.logits is not None:
            for pos in positions:
                low_logits_topk[pos] = _topk_logits(low.logits[0, pos, :], config.logits_topk)
                full_logits_topk[pos] = _topk_logits(full.logits[0, pos, :], config.logits_topk)

        for layer in layers:
            low_hidden = runner.normalize_hidden_state(low.hidden_states[layer + 1])[0].detach().cpu().float()
            full_hidden = runner.normalize_hidden_state(full.hidden_states[layer + 1])[0].detach().cpu().float()
            kv_ready = config.capture_kv and low.past_key_values is not None and full.past_key_values is not None

            for pos in positions:
                sample: Dict[str, Any] = {
                    "prompt_id": str(prompt_id),
                    "layer": int(layer),
                    "token_position": int(pos),
                    "residual_low": low_hidden[pos],
                    "residual_full": full_hidden[pos],
                }

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
                    low_k = low_k.detach().cpu().float()
                    low_v = low_v.detach().cpu().float()
                    full_k = full_k.detach().cpu().float()
                    full_v = full_v.detach().cpu().float()

                    sample["k_low"] = low_k
                    sample["v_low"] = low_v
                    sample["k_full"] = full_k
                    sample["v_full"] = full_v

                    if first_kv_shape is None:
                        first_kv_shape = tuple(low_k.shape)

                if config.capture_logits and pos in low_logits_topk:
                    low_vals, low_idx = low_logits_topk[pos]
                    full_vals, full_idx = full_logits_topk[pos]
                    low_on_full = low.logits[0, pos, :].detach().cpu().float()[full_idx]
                    sample["low_logits_values"] = low_vals
                    sample["low_logits_indices"] = low_idx
                    sample["full_logits_values"] = full_vals
                    sample["full_logits_indices"] = full_idx
                    sample["low_on_full_logits_values"] = low_on_full

                writer.add(sample)

    head_dim = 0
    num_heads = 0
    if first_kv_shape is not None:
        num_heads, head_dim = int(first_kv_shape[0]), int(first_kv_shape[1])

    manifest = writer.finalize(
        {
            "capture_kv": config.capture_kv,
            "capture_logits": config.capture_logits,
            "logits_topk": config.logits_topk,
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "model_name": config.model_name,
        }
    )
    return manifest
