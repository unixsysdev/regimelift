# RegimeLift

This repository is now centered on **RegimeLift**: a same-parent-model regime-uplift program for nested Gemma 3n operating modes. The active implementation and experiment track lives under `helmas3n/`.

Gemma 3n publishes E2B and E4B as separate model artifacts, but RegimeLift treats them as nested operating regimes of one parent family: a reduced E2B-like regime inside E4B and a fuller E4B regime. The project is therefore about regime handoff inside a shared MatFormer-style model family, not transfer between unrelated checkpoints.

Primary question:

> Can a low-activation nested regime inside Gemma 3n E4B be uplifted into the full-activation regime well enough to recover continuation behavior after handoff?

The earlier **He-LMAS** heterogeneous KV-bridge project is preserved under `archive/` for historical context and is no longer the active development target.

## Current status

Completed evidence so far:
- The low/full regime separation is real.
- Identity control matches proper low-to-full no-patch exactly.
- The first robust learned handoff site is `layer34,last1`.
- `layer16,last1` was a false lead: useful in early pilots, not robust on held-out data.
- Broader training washed out the site-specific signal.
- The oracle rows are reference patches, not ceilings.
- Completed heldout80: `targeted_mlp(layer34,last1)` reached `h8=0.1156`, `h16=0.0641` vs no-patch `h8=0.0172`, `h16=0.0133`.

Latest validation artifact:
- `helmas3n/artifacts/reports/targeted_site_study_v5_holdout80/` is the completed heldout80 report.

## Pending work (current phase)

The project is in a fixed-site objective-ablation phase at `layer34,last1`.

In progress now:
- Objective ablation run on heldout80:
  - output: `helmas3n/artifacts/reports/objective_ablation_layer34_last1_holdout80/`
  - variants: `state_only`, `state_logit`, `heavy_logit`, `short_horizon`
  - horizons: `h1,h4,h8,h16`

Immediate gates after ablation completes:
- Confirm winner beats the current `targeted_mlp(layer34,last1)` baseline at `h8` and `h16` on heldout80.
- Confirm winner improves over no-patch and keeps meaningful closure to `oracle_layer34` reference.

Queued support tracks after the winner check:
- Learned-vs-reference diagnostics:
  - script: `helmas3n/scripts/analyze_reference_vs_learned.py`
  - output: `helmas3n/artifacts/reports/reference_vs_learned_layer34_last1/`
- Cost/latency table:
  - script: `helmas3n/scripts/measure_handoff_costs.py`
  - output: `helmas3n/artifacts/reports/cost_table_layer34_last1/`

Then:
- Regenerate paper assets and PDF.
- Update results tables/figures in `paper/` and `helmas3n/experiments/`.
- Push synced artifacts and let CI publish the rolling paper release.

## Where to start reading

- [Current RegimeLift Gemma 3n track README](helmas3n/README.md)
- [RegimeLift experiment index](helmas3n/experiments/README.md)
- [RegimeLift targeted site study v1](helmas3n/experiments/targeted_site_study_v1/README.md)
- [Archived legacy He-LMAS README](archive/legacy-he-lmas/README.md)
- [Paper draft and build instructions](paper/README.md)

## Repository layout

- `helmas3n/`: active code, configs, scripts, tests, and live artifacts for the RegimeLift Gemma 3n track.
- `archive/`: frozen historical material for legacy He-LMAS only.
- `archive/legacy-he-lmas/code/`: relocated root-level legacy code (`configs/`, `scripts/`, `src/`, `tests/`).
- `paper/`: TeX source, generated figures, and the CI build path for the RegimeLift paper draft.
