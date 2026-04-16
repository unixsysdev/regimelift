# RegimeLift Gemma 3n Track

Nested-state uplift for Gemma 3n E4B.

## Thesis

Gemma 3n is published as separate E2B and E4B artifacts, but we treat them here as nested operating regimes of one parent model family. RegimeLift learns a minimal transport map that upgrades internal state from the reduced E2B-like regime inside Gemma 3n E4B into the corresponding fuller E4B regime, so decoding can resume closer to native full-regime behavior.

This track is intentionally different from original He-LMAS:

- Original He-LMAS: heterogeneous cross-model transfer.
- RegimeLift: same-parent-model regime-to-regime uplift.

## Scope

### In scope

- Paired state extraction from low/full activation regimes on identical prefixes.
- These regimes are different operating modes of the same parent family, not unrelated checkpoints.
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
  experiments/
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

## Latest Results

- Completed heldout80 targeted-site rerun: `artifacts/reports/targeted_site_study_v5_holdout80/`.
- `targeted_mlp(layer34,last1)` is the strongest learned method:
  - `h1=0.7750`, `h4=0.2031`, `h8=0.1156`, `h16=0.0641`
  - no-patch baseline: `h1=0.0375`, `h4=0.0250`, `h8=0.0172`, `h16=0.0133`
- `targeted_mlp(layer16,last1)` does not generalize on heldout80.
- Completed fixed-site objective ablation on heldout80:
  - output: `artifacts/reports/objective_ablation_layer34_last1_holdout80/`
  - winner: `residual_uplift_layer34_last1_short_horizon`
  - winner metrics: `h1=0.9125`, `h4=0.2375`, `h8=0.1328`, `h16=0.0742`
  - winner deltas: `delta_h8=+0.1016`, `delta_h16=+0.0484`
- Completed learned-vs-reference diagnostics:
  - output: `artifacts/reports/reference_vs_learned_layer34_last1/`
  - summary: no-patch KL `10.3642`, reference KL `1.1980`, learned KL `2.6606`
- Completed cost table support track:
  - output: `artifacts/reports/cost_table_layer34_last1/`
  - summary: mean low prefill `283.3 ms`, full restart `3191.4 ms`, pipeline (`low + handoff`) `3504.5 ms`

## Next queue

- Intermediate-regime transfer with fixed winner setting (`layer34,last1` + `short_horizon` objective).
- Larger held-out diversity beyond the current structured heldout80 pool.
- Lower-overhead systems evaluation for serving relevance (trigger policy + cache-transfer accounting).

## Notes

- `configs/extract.yaml` now defines non-identical low/full regime controls via `text_overrides`.
- Gemma loading defaults to text-only mode (`model.text_only: true`) to keep pilot runs focused on language-state uplift and avoid multimodal dependency overhead.
- For KV uplift at Gemma-scale dimensions, avoid `linear` unless intentionally testing an impractical upper-capacity baseline; prefer `lowrank`/`mlp`.
- `scripts/train_uplift.py` supports null baselines: `identity`, `mean_delta`, and `global_linear`.
