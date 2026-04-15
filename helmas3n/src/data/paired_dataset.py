from __future__ import annotations

import bisect
from collections import OrderedDict
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


@dataclass
class PairedDatasetInfo:
    num_samples: int
    num_layers: int
    hidden_dim: int
    capture_kv: bool
    capture_logits: bool
    logits_topk: int


class PairedStateDataset(Dataset):
    """Lazy shard loader for paired low/full regime state samples."""

    def __init__(self, root_dir: str | Path, target: str = "residual", shard_cache_size: int = 4) -> None:
        self.root_dir = Path(root_dir)
        self.target = target
        self.shard_cache_size = max(1, int(shard_cache_size))
        if self.target not in {"residual", "kv"}:
            raise ValueError(f"Unsupported target: {target}")

        manifest_path = self.root_dir / "manifest.json"
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text())
            shards = self.manifest.get("shards", [])
            self.shards = [self.root_dir / s["path"] for s in shards]
            self.shard_sizes = [int(s["num_samples"]) for s in shards]
        else:
            self.shards = sorted(self.root_dir.glob("shard_*.pt"))
            self.shard_sizes = []
            self.manifest = {
                "num_layers": 0,
                "hidden_dim": 0,
                "capture_kv": False,
                "capture_logits": False,
                "logits_topk": 0,
            }
            for shard in self.shards:
                data = torch.load(shard, map_location="cpu")
                self.shard_sizes.append(int(data["layer"].numel()))

        if not self.shards:
            raise FileNotFoundError(f"No shards found in {self.root_dir}")

        self.cum_sizes: List[int] = []
        running = 0
        for size in self.shard_sizes:
            running += size
            self.cum_sizes.append(running)

        self._shard_cache: OrderedDict[int, Dict[str, object]] = OrderedDict()

    @property
    def info(self) -> PairedDatasetInfo:
        num_samples = self.cum_sizes[-1]
        return PairedDatasetInfo(
            num_samples=num_samples,
            num_layers=int(self.manifest.get("num_layers", 0)),
            hidden_dim=int(self.manifest.get("hidden_dim", 0)),
            capture_kv=bool(self.manifest.get("capture_kv", False)),
            capture_logits=bool(self.manifest.get("capture_logits", False)),
            logits_topk=int(self.manifest.get("logits_topk", 0)),
        )

    def __len__(self) -> int:
        return self.cum_sizes[-1]

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_idx = bisect.bisect_right(self.cum_sizes, index)
        prev = 0 if shard_idx == 0 else self.cum_sizes[shard_idx - 1]
        local_idx = index - prev
        return shard_idx, local_idx

    def _load_shard(self, shard_idx: int) -> Dict[str, object]:
        if shard_idx in self._shard_cache:
            data = self._shard_cache.pop(shard_idx)
            self._shard_cache[shard_idx] = data
            return data

        data = torch.load(self.shards[shard_idx], map_location="cpu")
        self._shard_cache[shard_idx] = data
        while len(self._shard_cache) > self.shard_cache_size:
            self._shard_cache.popitem(last=False)
        return data

    def __getitem__(self, index: int) -> Dict[str, object]:
        shard_idx, local_idx = self._locate(index)
        shard = self._load_shard(shard_idx)

        sample: Dict[str, object] = {
            "prompt_id": shard["prompt_id"][local_idx],
            "layer": shard["layer"][local_idx],
            "token_position": shard["token_position"][local_idx],
        }

        if self.target == "residual":
            sample["source_state"] = shard["residual_low"][local_idx]
            sample["target_state"] = shard["residual_full"][local_idx]
        else:
            k_low = shard["k_low"][local_idx].reshape(-1)
            v_low = shard["v_low"][local_idx].reshape(-1)
            k_full = shard["k_full"][local_idx].reshape(-1)
            v_full = shard["v_full"][local_idx].reshape(-1)
            sample["source_state"] = torch.cat([k_low, v_low], dim=0)
            sample["target_state"] = torch.cat([k_full, v_full], dim=0)
            sample["k_low"] = shard["k_low"][local_idx]
            sample["v_low"] = shard["v_low"][local_idx]
            sample["k_full"] = shard["k_full"][local_idx]
            sample["v_full"] = shard["v_full"][local_idx]

        if "full_logits_values" in shard:
            sample["full_logits_values"] = shard["full_logits_values"][local_idx]
            sample["full_logits_indices"] = shard["full_logits_indices"][local_idx]
        if "low_logits_values" in shard:
            sample["low_logits_values"] = shard["low_logits_values"][local_idx]
            sample["low_logits_indices"] = shard["low_logits_indices"][local_idx]
        if "low_on_full_logits_values" in shard:
            sample["low_on_full_logits_values"] = shard["low_on_full_logits_values"][local_idx]

        return sample


def paired_collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {
        "prompt_id": [b["prompt_id"] for b in batch],
        "layer": torch.stack([b["layer"] for b in batch]),
        "token_position": torch.stack([b["token_position"] for b in batch]),
        "source_state": torch.stack([b["source_state"] for b in batch]),
        "target_state": torch.stack([b["target_state"] for b in batch]),
    }

    optional_tensor_keys = [
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
    for key in optional_tensor_keys:
        if key in batch[0]:
            out[key] = torch.stack([b[key] for b in batch])

    return out
