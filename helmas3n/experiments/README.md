# RegimeLift Experiments

This directory is the reader-facing index for completed RegimeLift experiments.

- `targeted_site_study_v1/`: first completed targeted handoff study.
  - winner: `targeted_mlp(layer34,last1)` in the original targeted-site phase.

- `objective_ablation_layer34_last1_holdout80` (runtime artifact set):
  - path: `helmas3n/artifacts/reports/objective_ablation_layer34_last1_holdout80/`
  - winner: `residual_uplift_layer34_last1_short_horizon`
  - key gains: `delta_h8=+0.1016`, `delta_h16=+0.0484`

- `reference_vs_learned_layer34_last1` (runtime artifact set):
  - path: `helmas3n/artifacts/reports/reference_vs_learned_layer34_last1/`
  - note: reference patch has lower KL than learned patch; learned patch remains behaviorally useful in continuation metrics.

- `cost_table_layer34_last1` (runtime artifact set):
  - path: `helmas3n/artifacts/reports/cost_table_layer34_last1/`
  - note: current pipeline is slower than full restart in this instrumentation-heavy setup.

- `intermediate_full_targeted_site_study_v1` (runtime artifact set):
  - path: `helmas3n/artifacts/reports/intermediate_full_targeted_site_study_v1/`
  - result:
    - strong recovery at h8 (error → 0)
    - partial recovery at h16
  - note: confirms portability of layer34 intervention beyond minimum→full

Runtime outputs are written under `helmas3n/artifacts/reports/`.
