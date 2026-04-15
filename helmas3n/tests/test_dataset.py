from __future__ import annotations

import json
from pathlib import Path

import torch

from helmas3n.src.data.paired_dataset import PairedStateDataset, paired_collate


def test_dataset_loads_shards(tmp_path: Path) -> None:
    root = tmp_path / "paired"
    root.mkdir(parents=True)

    shard = {
        "prompt_id": ["a", "b"],
        "layer": torch.tensor([0, 1]),
        "token_position": torch.tensor([3, 4]),
        "residual_low": torch.randn(2, 8),
        "residual_full": torch.randn(2, 8),
        "full_logits_values": torch.randn(2, 5),
        "full_logits_indices": torch.randint(0, 50, (2, 5)),
        "low_logits_values": torch.randn(2, 5),
        "low_logits_indices": torch.randint(0, 50, (2, 5)),
    }
    torch.save(shard, root / "shard_00000.pt")
    manifest = {
        "num_layers": 2,
        "hidden_dim": 8,
        "capture_kv": False,
        "capture_logits": True,
        "logits_topk": 5,
        "shards": [{"path": "shard_00000.pt", "num_samples": 2}],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))

    ds = PairedStateDataset(root, target="residual")
    assert len(ds) == 2
    sample = ds[1]
    assert sample["source_state"].shape[-1] == 8

    batch = paired_collate([ds[0], ds[1]])
    assert batch["source_state"].shape == (2, 8)
