# Objective Ablation @ layer34,last1

- Variants: 4
- Horizons: h1, h4, h8, h16

| method | train loss mix | held-out h1 | held-out h4 | held-out h8 | held-out h16 | delta_h8 | delta_h16 | cost per prompt (ms, h16) | closure to reference-layer34 (h8) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| residual_uplift_layer34_last1_state_only | `mse=1.0;cos=0.2;kl=0.0;ce=0.0` | 0.312500 | 0.081250 | 0.048438 | 0.028906 | 0.017188 | 0.003125 | 8963.287 | 0.393 |
| residual_uplift_layer34_last1_state_logit | `mse=1.0;cos=0.2;kl=0.3;ce=0.0` | 0.800000 | 0.209375 | 0.118750 | 0.065625 | 0.087500 | 0.039844 | 9280.770 | 2.000 |
| residual_uplift_layer34_last1_heavy_logit | `mse=1.0;cos=0.2;kl=1.0;ce=0.0` | 0.862500 | 0.225000 | 0.126562 | 0.070312 | 0.095312 | 0.044531 | 8675.035 | 2.179 |
| residual_uplift_layer34_last1_short_horizon | `mse=1.0;cos=0.2;kl=0.3;ce=1.0` | 0.912500 | 0.237500 | 0.132812 | 0.074219 | 0.101562 | 0.048438 | 9003.797 | 2.321 |
