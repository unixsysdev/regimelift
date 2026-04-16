# Full-like vs Low-like Diagnostic

- prompts: 80
- layer: 34

| method | top1 match to full | top1 match to low | KL(full||method) | KL(low||method) |
|---|---:|---:|---:|---:|
| no_patch | 0.037500 | 0.000000 | 10.364227 | 15.930509 |
| reference_layer34 | 0.387500 | 0.037500 | 1.197973 | 9.800478 |
| broad_mlp | 0.062500 | 0.212500 | 7.455112 | 3.931631 |
| targeted_layer34 | 0.287500 | 0.025000 | 2.660585 | 4.552105 |
