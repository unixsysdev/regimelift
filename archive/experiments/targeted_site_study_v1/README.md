# Frozen Targeted Site Study v1

This directory is the frozen result set for the first completed HeLMAS-3n targeted site study.

## What it established

- Identity control matched proper low-to-full no-patch exactly.
- `layer16,last1` was not robust enough to carry forward.
- `layer34,last1` was the first learned handoff site that generalized on the held-out slice.
- The broad MLP baseline preserved some signal, but it washed out the strongest site-specific effect.
- Oracle rows should be treated as reference patches, not strict ceilings.

## Frozen outputs

- [Decision memo](decision_memo.md)
- [Targeted site report](targeted_site_report.md)
- [Targeted site rows](targeted_site_rows.csv)
- [Targeted site summary](targeted_site_summary.json)

## Why this folder exists

The paper now uses this result as the starting point for the next-stage validation, rather than as the end of the project.
