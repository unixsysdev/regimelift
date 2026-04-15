# Targeted Site Study Decision Memo

## Current conclusion

`layer34,last1` is the first learned handoff site that is both strong and usable.

`layer16,last1` is not robust enough to carry forward.

The control path is sound because `identity` matches `low_to_full_no_patch` exactly on the evaluated splits.

## Evidence

- Pilot:
  - `layer34,last1` reaches strong continuation recovery.
  - `layer16,last1` does not beat the broad baseline consistently.
- Held-out:
  - `layer34,last1` still generalizes.
  - `layer16,last1` collapses.
- Broad MLP:
  - broad training preserved some signal, but it washed out the useful site-specific effect.

## Oracle interpretation

Do not call `oracle_layer34` a ceiling.

Use one of these labels instead:

- same-site exact-substitution reference
- reference patch

The learned `layer34,last1` row beating the oracle on some horizons means the oracle row is not a strict behavioral upper bound in this setup.

## Next gate

Keep `layer34,last1` frozen as the primary learned target.

Expand held-out evaluation.

Keep `broad_mlp` in every table.

Do not change the loss, KV target, or flow before the larger held-out run.
