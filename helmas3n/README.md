# HeLMAS-3n

Nested-state uplift for Gemma 3n E4B.

## Thesis

HeLMAS-3n learns a minimal transport map that upgrades internal state from a low-activation nested regime of **the same Gemma 3n E4B checkpoint** into the corresponding full-activation regime, so decoding can resume closer to native full-regime behavior.

This is intentionally different from original He-LMAS:

- Original He-LMAS: heterogeneous cross-model transfer.
- HeLMAS-3n: same-parent-model regime-to-regime uplift.

## Scope

### In scope

- Paired state extraction from low/full activation regimes on identical prefixes.
- Residual uplift baselines: linear, low-rank residual, small MLP.
- KV uplift baseline for direct cache handoff comparison.
- Handoff-focused evaluation hierarchy:
  - next-token KL to full regime
  - exact next-token agreement
  - short continuation match rate
  - long-horizon drift after handoff

### Out of scope

- Cross-family transfer.
- E2B to E4B checkpoint transfer.
- Flow-first training.

A conditional flow module is included only as a phase-2 refinement placeholder.

## Project Layout

```
helmas3n/
  configs/
  scripts/
  src/
    data/
    models/
    losses/
    gemma/
    eval/
  tests/
  artifacts/
```

## Quick Start

```bash
cd helmas3n
pip install -r ../requirements.txt

# 1) Collect paired states
python scripts/collect_paired_states.py --config configs/extract.yaml

# 1.5) Verify low/full regimes are meaningfully different
python scripts/sanity_regime_report.py --config configs/extract.yaml --max-prompts 50

# 2) Train residual uplift baseline
python scripts/train_uplift.py --config configs/train_residual.yaml

# 2a) Null controls
python scripts/train_uplift.py --config configs/train_residual_identity.yaml
python scripts/train_uplift.py --config configs/train_residual_mean_delta.yaml
python scripts/train_uplift.py --config configs/train_residual_global_linear.yaml

# 3) Train KV uplift baseline
python scripts/train_uplift.py --config configs/train_kv.yaml

# 4) Evaluate handoff/alignment
python scripts/eval_handoff.py --config configs/train_residual.yaml
python scripts/analyze_alignment.py --config configs/extract.yaml

# 5) Run residual pilot suite (null controls + learned baselines)
python scripts/run_residual_pilot.py --extract-config configs/extract.yaml
```

## Data Format

Extractor output is a reproducible sharded tensor dataset:

- `artifacts/paired_states/manifest.json`
- `artifacts/paired_states/shard_00000.pt`, ...

Each sample stores:

- `prompt_id`, `token_position`, `layer`
- `residual_low`, `residual_full`
- optional `k_low`, `v_low`, `k_full`, `v_full`
- optional sparse logits (`*_logits_values`, `*_logits_indices`)

## Milestones Implemented

1. Paired-state extraction pipeline.
2. Residual uplift baselines + state/logit losses.
3. KV uplift baseline + direct comparison support.
4. Flow refinement placeholder module (not default).

## Notes

- `configs/extract.yaml` now defines non-identical low/full regime controls via `text_overrides`.
- Gemma loading defaults to text-only mode (`model.text_only: true`) to keep pilot runs focused on language-state uplift and avoid multimodal dependency overhead.
- For KV uplift at Gemma-scale dimensions, avoid `linear` unless intentionally testing an impractical upper-capacity baseline; prefer `lowrank`/`mlp`.
- `scripts/train_uplift.py` supports null baselines: `identity`, `mean_delta`, and `global_linear`.
