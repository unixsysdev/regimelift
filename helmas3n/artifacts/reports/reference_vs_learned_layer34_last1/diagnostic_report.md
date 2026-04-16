# Learned vs Reference Patch Diagnostics

- prompts: 80
- layer: 34

| method | mean KL(full||method) | top1 match to full | final hidden cosine to full |
|---|---:|---:|---:|
| no_patch | 10.364227 | 0.037500 | 0.353990 |
| reference_patch | 1.197973 | 0.387500 | 0.353990 |
| learned_patch | 2.660585 | 0.287500 | 0.353990 |
