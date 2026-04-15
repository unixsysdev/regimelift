# He-LMAS / HeLMAS-3n

He-LMAS started as a heterogeneous KV-cache bridge project: transfer reasoning across different model families by projecting teacher cache/state into a smaller student. HeLMAS-3n is the next stage: same-parent-model regime uplift inside **Gemma 3n E4B**, where the task is to lift a low-activation nested state into the full-activation regime of the same checkpoint.

## Two attempts, one repository

### Legacy starting point: He-LMAS
The original He-LMAS project explored cross-model transfer between different Qwen checkpoints. It focused on RoPE-aware KV projection, layer blending, and attention-consistency-style objectives.

That work is now archived here:
- [Legacy He-LMAS overview](archive/legacy-he-lmas/README.md)
- [Frozen targeted experiment v1](archive/experiments/targeted_site_study_v1/README.md)

### Current program: HeLMAS-3n
HeLMAS-3n asks a narrower and more falsifiable question:

> Can a low-activation nested regime inside Gemma 3n E4B be uplifted into the full-activation regime well enough to recover continuation behavior after handoff?

This is not cross-family cache porting. It is regime-to-regime transport inside one parent model.

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
- That run is intentionally isolated and uses absolute paths so it can be audited from another session.

## Where to start reading

- [Current HeLMAS-3n technical README](helmas3n/README.md)
- [Archived legacy He-LMAS README](archive/legacy-he-lmas/README.md)
- [Frozen targeted site study v1](archive/experiments/targeted_site_study_v1/README.md)
- [Paper draft and build instructions](paper/README.md)

## Repository layout

- `helmas3n/`: active code, configs, scripts, tests, and live artifacts for the Gemma 3n program.
- `archive/`: frozen historical material and completed experiment snapshots.
- `paper/`: TeX source, generated figures, and the CI build path for the research draft.

## Practical note

The current scripts use absolute-path resolution for the evaluator runs. That is deliberate: it makes the live held-out rerun reproducible and avoids the path bugs that previously truncated the held-out split.
