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

## Latest Results

- Completed heldout80 targeted-site rerun: `artifacts/reports/targeted_site_study_v5_holdout80/`.
- `targeted_mlp(layer34,last1)` is the strongest learned method:
  - `h1=0.7750`, `h4=0.2031`, `h8=0.1156`, `h16=0.0641`
  - no-patch baseline: `h1=0.0375`, `h4=0.0250`, `h8=0.0172`, `h16=0.0133`
- Completed fixed-site objective ablation on heldout80:
  - winner: `residual_uplift_layer34_last1_short_horizon`
  - metrics: `h1=0.9125`, `h4=0.2375`, `h8=0.1328`, `h16=0.0742`
- Intermediate → full transfer (targeted site study):
  - path: `artifacts/reports/intermediate_full_targeted_site_study_v1/`
  - `layer34,last1` remains effective
  - h8: full recovery (error → 0)
  - h16: partial recovery
  - broad MLP and identity: no improvement

## Next queue

- Complete regime triangle with `minimum -> intermediate`.
- Expand held-out diversity beyond structured heldout80.
- Improve systems evaluation (latency + serving policy).

## Notes

- `configs/extract.yaml` defines non-identical low/full regime controls via `text_overrides`.
- Gemma loading defaults to text-only mode.
- Prefer `lowrank`/`mlp` over `linear` for KV uplift at scale.
- `scripts/train_uplift.py` supports null baselines: `identity`, `mean_delta`, `global_linear`.
