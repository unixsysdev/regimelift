# He-LMAS: Heterogeneous Latent Manifold Alignment System

> **Core Thesis**: The KV Cache is a fossilized record of causal history. We inject a tensor that *looks like* it was produced by the small model, but encodes the big model's intelligence.

## Overview

He-LMAS enables **lossless reasoning transfer** between heterogeneous LLMs (e.g., Qwen3-8B → Qwen3-1.7B) through RoPE-aware KV cache projection. Instead of text-based handoffs, we surgically transplant the Teacher's "thought process" into the Student's attention memory.

## V2 Architecture

```
Teacher (Qwen3-8B, 36 layers)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                     He-LMAS Bridge V2                          │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  1. LAYER BLENDING (36 → 28 layers)                           │
│     ┌────────────────┬─────────────────┬─────────────────┐    │
│     │     skip       │     blend       │   attention     │    │
│     │ Uniform stride │  Conv1d local   │  Global pooling │    │
│     │   (drops 8)    │   (~10K params) │   (~2K params)  │    │
│     └────────────────┴─────────────────┴─────────────────┘    │
│                           │                                    │
│  2. DE-ROTATE ROPE ───────┼── (Full mode: undo θ=1M)          │
│                           │                                    │
│  3. DEEP PROJECTOR ───────┼── Linear → GELU → Linear          │
│     └── ~3.6M params      │   (configurable depth)            │
│                           │                                    │
│  4. RE-ROTATE ROPE ───────┼── (Full mode: apply θ=1M)         │
│                           ▼                                    │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
Student (Qwen3-1.7B, 28 layers) + Injected Reasoning
```

**Total trainable parameters: ~3.6M** (tiny compared to the LLMs)

## Key Innovation: Attention Consistency Loss

Instead of forcing KV caches to match directly (geometrically impossible), we force the **Attention Output** to match:

```
L = ||Attn(Q, Φ(KV_Teacher)) - Attn(Q, KV_Student^TeacherForced)||²
```

The Student learns to see the same "Context Vector" when looking at projected Teacher memory as it would if it had computed the memory itself.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run baseline evaluation (see where 1.7B fails)
python scripts/baseline_eval.py

# Train the bridge projector (H200 optimized)
python scripts/train_bridge.py --config configs/h200.yaml --dataset gsm8k

# Run inference demo
python scripts/inference_demo.py --checkpoint checkpoints/bridge_final.pt
```

## Configuration Options

### Layer Blending

```yaml
bridge:
  layer_blending: "attention"  # Options: skip, blend, attention
  blend_kernel_size: 3         # For blend mode only
```

| Mode | Description | Params |
|------|-------------|--------|
| `skip` | Uniform stride selection (fast baseline) | 0 |
| `blend` | Conv1d smooth blending | ~10K |
| `attention` | Global attention pooling (default) | ~2K |

### Projector Depth

```yaml
bridge:
  projector_depth: 2    # 1=shallow (linear), 2+=deep (with GELU)
  hidden_expansion: 2.0 # Hidden dim = head_dim × expansion
```

### RoPE Handling

```yaml
bridge:
  rope_handling: "full"       # "naive" (skip) or "full" (de-rotate/re-rotate)
  teacher_rope_base: 1000000.0
  student_rope_base: 1000000.0
```

## Validation

The training loop includes automatic validation on held-out samples:

```yaml
training:
  eval_every: 1000    # Run validation every N steps
  eval_samples: 50    # Number of samples to evaluate
```

Validation compares:
- **With Bridge**: Student accuracy using injected Teacher reasoning
- **Without Bridge**: Student-alone baseline

## Key Components

- **`src/bridge.py`**: V2 Bridge with layer blending, deep projectors, full RoPE
- **`src/training.py`**: Attention Consistency Loss + validation loop
- **`src/rope_utils.py`**: RoPE rotation/de-rotation utilities
- **`src/kv_cache_utils.py`**: KV cache extraction and injection
- **`src/data_loader.py`**: HuggingFace dataset integration (GSM8K, MATH, etc.)

## H200 Optimizations

The framework includes specific optimizations for NVIDIA H200:

- **Flash Attention 2**: Faster attention computation
- **BF16 Precision**: Optimal tensor core usage
- **AMP Training**: Mixed precision with GradScaler
- **torch.compile()**: Kernel fusion and optimization

See `configs/h200.yaml` for the full H200 configuration.

## Datasets Supported

- **GSM8K**: Grade school math (8.5K problems)
- **OpenMathInstruct-2**: Large scale math instruction (5.75M)
- **MetaMathQA**: Math question answering
- **MATH**: Competition mathematics

## References

- **LatentMAS**: [arXiv:2511.20639](https://arxiv.org/abs/2511.20639)
- **Qwen3**: [HuggingFace Collection](https://huggingface.co/Qwen)

## License

MIT
