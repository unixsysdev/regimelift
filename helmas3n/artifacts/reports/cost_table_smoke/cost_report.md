# Cost Table @ layer34,last1

- prompts: 1
- decode horizon: 1

| Component | Mean ms/prompt |
|---|---:|
| low prefill | 615.870 |
| full restart (prefill+decode) | 254.881 |
| handoff resume (patched prefill+decode) | 309.549 |
| pipeline: low prefill + handoff resume | 925.419 |

- pipeline minus full restart: 670.539 ms
- h16 continuation match: uplift=1.000000, low=0.000000
