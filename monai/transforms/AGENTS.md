# Transforms

## Layout

Each topic folder (`croppad`, `intensity`, `io`, `post`, `signal`, and the
others) splits into `array.py` and `dictionary.py`. The array class works on a
single array or `MetaTensor` (for example `LoadImage` in `io/array.py`); the
dictionary class applies the same operation per key of a dict and usually ends
in `d` (`LoadImaged` in `io/dictionary.py`). A fix for a transform almost
always needs both variants, plus their tests.

## Things that are easy to get wrong

- Random transforms derive from `RandomizableTransform` and draw from
  `self.R`, set through `set_random_state`. Never use the global `random` or
  `numpy.random` inside a transform: `Compose` and the data loaders seed each
  transform per sample, and global state breaks reproducibility and worker
  safety.
- Transforms run in multiple data loader worker processes.
  `RandomizableTransform` may store sampled parameters on `self` for the
  current call or for explicit `randomize=False` reuse. Do not share one
  transform instance across concurrent calls or rely on prior-call state
  unless the API explicitly requests reuse.
- Many transforms are invertible and record what they did through
  `MetaTensor` applied operations and `InvertibleTransform` tracing. If you
  change a transform's parameters or geometry handling, check the inverse
  path (`monai/transforms/inverse.py`) and `Invertd` tests still make sense.
- Lazy transforms live in `lazy/`; changes to resampling behavior can affect
  both the eager and the lazy execution of the same transform. Read
  `docs/source/lazy_resampling.rst` before touching resample logic.
- Dictionary transforms take `keys` and share one code path through
  `MapTransform`; missing-key behavior is controlled by `allow_missing_keys`,
  do not invent a new flag for it.
- New transforms must be exported from the topic folder and from
  `monai/transforms/__init__.py`, and listed in `docs/source/transforms.rst`.
