# He-LMAS: Heterogeneous Latent Manifold Alignment System

> **Core Thesis**: The KV Cache is a fossilized record of causal history. We inject a tensor that *looks like* it was produced by the small model, but encodes the big model's intelligence.

## Overview

He-LMAS enables **lossless reasoning transfer** between heterogeneous LLMs (e.g., Qwen3-8B → Qwen3-1.7B) through RoPE-aware KV cache projection. Instead of text-based handoffs, we surgically transplant the Teacher's "thought process" into the Student's attention memory.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        He-LMAS Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Teacher    │    │    Bridge    │    │   Student    │      │
│  │  (Qwen3-8B)  │───▶│   Projector  │───▶│ (Qwen3-1.7B) │      │
│  │              │    │              │    │              │      │
│  │  36 Layers   │    │  36 → 28     │    │  28 Layers   │      │
│  │  8 KV Heads  │    │  Dimension   │    │  8 KV Heads  │      │
│  │  4096 dim    │    │  Projection  │    │  2048 dim    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                                       │               │
│         ▼                                       ▼               │
│   KV Cache (Thinking)              Injected KV + Answer         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run baseline evaluation (see where 1.7B fails)
python scripts/baseline_eval.py

# Train the bridge projector
python scripts/train_bridge.py --config configs/default.yaml

# Run inference demo
python scripts/inference_demo.py --checkpoint checkpoints/bridge_best.pt
```

## Key Components

- **`src/bridge.py`**: The Heterogeneous Bridge (manifold projector)
- **`src/rope_utils.py`**: RoPE rotation/de-rotation utilities
- **`src/kv_cache_utils.py`**: KV cache extraction and injection
- **`src/training.py`**: Online streaming training loop

## References

- **LatentMAS**: [arXiv:2511.20639](https://arxiv.org/abs/2511.20639)
- **Qwen3**: [HuggingFace Collection](https://huggingface.co/Qwen)

## License

MIT
