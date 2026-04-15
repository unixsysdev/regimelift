# HeLMAS-3n (Active) / He-LMAS (Archived)

This repository is now centered on **HeLMAS-3n**: same-parent-model regime uplift inside Gemma 3n E4B.

Gemma 3n publishes E2B and E4B as separate model artifacts, but the active research program here treats them as nested operating regimes of one parent family: a reduced E2B-like regime inside E4B and a fuller E4B regime. The project is therefore about regime handoff inside a shared MatFormer-style model family, not transfer between unrelated checkpoints.

Primary question:

> Can a low-activation nested regime inside Gemma 3n E4B be uplifted into the full-activation regime well enough to recover continuation behavior after handoff?

The earlier **He-LMAS** heterogeneous KV-bridge project is preserved for historical context under `archive/` and is no longer the active development target.

## Current status

Completed evidence so far:
- The low/full regime separation is real.
- Identity control matches proper low-to-full no-patch exactly.
- The first robust learned handoff site is `layer34,last1`.
- `layer16,last1` was a false lead: useful in early pilots, not robust on held-out data.
- Broader training washed out the site-specific signal.
- The oracle rows are reference patches, not ceilings.

Active validation:
- `helmas3n/artifacts/reports/targeted_site_study_v5_holdout80/` is the current held-out80 rerun.

## Where to start reading

- [Current HeLMAS-3n technical README](helmas3n/README.md)
- [HeLMAS-3n experiment index](helmas3n/experiments/README.md)
- [HeLMAS-3n targeted site study v1](helmas3n/experiments/targeted_site_study_v1/README.md)
- [Archived legacy He-LMAS README](archive/legacy-he-lmas/README.md)
- [Paper draft and build instructions](paper/README.md)

## Repository layout

- `helmas3n/`: active code, configs, scripts, tests, and live artifacts for the Gemma 3n program.
- `archive/`: frozen historical material for legacy He-LMAS only.
- `archive/legacy-he-lmas/code/`: relocated root-level legacy code (`configs/`, `scripts/`, `src/`, `tests/`).
- `paper/`: TeX source, generated figures, and the CI build path for the research draft.
