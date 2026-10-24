# Engines and handlers

`Trainer` and `Evaluator` (in `trainer.py` and `evaluator.py`) build on the
`Workflow` base class and on PyTorch Ignite. Handlers live in
`monai/handlers` and are attached to an engine to add checkpointing,
logging, early stopping, and metrics output.

## Things that are easy to get wrong

- Ignite is an optional dependency here: it is imported through
  `monai.utils.optional_import` with the minimum version pinned in
  `IgniteInfo.OPT_IMPORT_VERSION` (`monai/utils/enums.py`). Never import
  `ignite` directly at module top level in this package.
- Handlers react to Ignite events (`Events.EPOCH_COMPLETED`, and so on) and
  read from `state.output`. Changing what a trainer puts into `state.output`
  breaks handlers in another package; keep the documented keys stable.
- Metrics handlers expect the metric lifecycle from
  `monai/metrics/AGENTS.md`: accumulate per iteration, `aggregate()` and
  `reset()` at event time. Do not call `aggregate()` per batch in a handler.
- `monai/engines` decides when things happen; `monai/handlers` decides what
  happens. A new capability usually means a small handler plus, at most, a
  new event or state entry in the engine, not logic inside the training
  loop.
