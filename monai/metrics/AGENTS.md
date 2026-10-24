# Metrics

Shared conventions with `monai/losses` (see `monai/losses/AGENTS.md`):
`include_background` defaults to `True`, and reductions come from
`MetricReduction` in `monai.utils`.

## How metrics accumulate

Most metrics derive from `Cumulative` or `CumulativeIterationMetric` in
`metric.py`. Calls during an epoch append to internal buffers with
`__call__` / `update`; `aggregate()` computes the result from the buffers;
`reset()` clears them.

## Things that are easy to get wrong

- Forgetting `reset()` between epochs or evaluations is the classic metrics
  bug: the second result silently includes the first run's data. Any new
  metric must reset every buffer it owns, and handlers in `monai/handlers`
  call `reset()` and `aggregate()` at event time, not per batch.
- `aggregate()` must not destroy the buffers unless documented; some
  workflows aggregate intermediate results and keep accumulating.
- Handle the not-applicable case (for example, a class missing from both
  prediction and target) the way the neighboring metric does, usually by
  producing `nan` and letting `get_not_nans` or the handler filter it,
  instead of returning 0 and flattering the score.
- Distance-based metrics (Hausdorff, surface distance) are computed per
  class and can be the slow part of a test suite; keep test inputs small
  and mark slow tests as described in `tests/AGENTS.md`.
