# Cost Table @ layer34,last1

- prompts: 80
- decode horizon: 16

| Component | Mean ms/prompt |
|---|---:|
| low prefill | 283.289 |
| full restart (prefill+decode) | 3191.393 |
| handoff resume (patched prefill+decode) | 3221.192 |
| pipeline: low prefill + handoff resume | 3504.481 |

- pipeline minus full restart: 313.089 ms
- h16 continuation match: uplift=0.074219, low=0.025781
