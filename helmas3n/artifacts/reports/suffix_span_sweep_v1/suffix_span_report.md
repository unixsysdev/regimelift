# Suffix Span Sweep Report

- Prompt rows: 1440
- Summary rows: 36
- Identity control passed: true

## Best Oracle Rows
| Horizon | Site | Span | Mean match | Mean no-patch | Mean delta |
|---|---:|---:|---:|---:|---:|
| h1 | layer34 | last1 | 0.250000 | 0.000000 | 0.250000 |
| h4 | stride | last1 | 0.081250 | 0.006250 | 0.075000 |
| h8 | stride | last1 | 0.046875 | 0.009375 | 0.037500 |
| h16 | stride | last1 | 0.031250 | 0.012500 | 0.018750 |

## Best MLP Rows
| Horizon | Site | Span | Mean match | Mean no-patch | Mean delta | Mean closure |
|---|---:|---:|---:|---:|---:|---:|
| h1 | layer16 | last1 | 0.000000 | 0.000000 | 0.000000 | 0.000 |
| h4 | layer16 | last1 | 0.012500 | 0.006250 | 0.006250 | 1.000 |
| h8 | layer16 | last1 | 0.021875 | 0.009375 | 0.012500 | 1.000 |
| h16 | layer16 | last1 | 0.017188 | 0.012500 | 0.004687 | 1.000 |

## Span Effect
- layer34: best oracle h8 is last1 (delta 0.028125); best oracle h16 is last1 (delta 0.015625).
- layer16: best oracle h8 is last1 (delta 0.012500); best oracle h16 is last1 (delta 0.004687).
- stride: best oracle h8 is last1 (delta 0.037500); best oracle h16 is last1 (delta 0.018750).

## Files
- [Prompt rows](./suffix_span_prompt_rows.csv)
- [Summary rows](./suffix_span_summary.csv)
- [Summary JSON](./suffix_span_summary.json)
