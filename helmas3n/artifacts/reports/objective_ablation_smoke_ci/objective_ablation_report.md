# Objective Ablation @ layer34,last1

- Variants: 4
- Horizons: h1

| method | train loss mix | held-out h1 | held-out h4 | held-out h8 | held-out h16 | delta_h8 | delta_h16 | cost per prompt (ms, h16) | closure to reference-layer34 (h8) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| residual_uplift_layer34_last1_state_only | `mse=1.0;cos=0.2;kl=0.0;ce=0.0` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -- | -- |
| residual_uplift_layer34_last1_state_logit | `mse=1.0;cos=0.2;kl=0.3;ce=0.0` | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -- | -- |
| residual_uplift_layer34_last1_heavy_logit | `mse=1.0;cos=0.2;kl=1.0;ce=0.0` | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -- | -- |
| residual_uplift_layer34_last1_short_horizon | `mse=1.0;cos=0.2;kl=0.3;ce=1.0` | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -- | -- |
