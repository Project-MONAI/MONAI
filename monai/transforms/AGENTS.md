# AGENTS.md — `monai/transforms`

Notes for AI coding assistants changing `monai/transforms`. The repository-wide rules live in the
root [AGENTS.md](../../AGENTS.md); human contributors should read
[CONTRIBUTING.md](../../CONTRIBUTING.md).

## Module map

| Path | Contents |
|------|----------|
| `transform.py` | base classes: `Transform`, `MapTransform`, `RandomizableTransform`, `LazyTransform` |
| `traits.py` | `LazyTrait`, `InvertibleTrait`, `RandomizableTrait`, `MultiSampleTrait`, `ReduceTrait` |
| `inverse.py` | `TraceableTransform`, `InvertibleTransform` — the transform stack and its inversion |
| `compose.py` | `Compose`, the pipeline that runs the transforms |
| `lazy/` | lazy resampling: `ApplyPending` and the pending `Affine`/`DisplacementField` bookkeeping |
| `croppad/`, `intensity/`, `io/`, `post/`, `regularization/`, `signal/`, `smooth_field/`, `spatial/`, `utility/` | the transform families |
| `meta_utility/` | dictionary-only transforms that work on `MetaTensor` metadata |
| `utils.py` | helpers shared by the transforms |

Each family directory follows the same convention:

- `array.py` — the transform applied to a single tensor, e.g. `Spacing`.
- `dictionary.py` — the dictionary variant named with a `d` suffix, e.g. `Spacingd`.
- `functional.py` — the functional API, where present (`croppad`, `lazy`, `spatial`).
- `batch.py` — batched dictionary transforms (`croppad` only).

## Traps

**A transform is usually three or four classes.** Adding a transform means the array class, the `*d`
dictionary class, and a `Rand*` class when the transform is randomizable. A dictionary transform
derives from `MapTransform` and selects the keys it applies to with `key_iterator`; it is not the
array class with a different name.

**`MetaTensor` must survive the call.** `MetaTensor` (`monai/data/meta_tensor.py`) carries the
affine matrix, the original spatial shape and the stack of applied transforms. Converting to a
plain `torch.Tensor` or to numpy inside a transform silently drops that metadata and breaks
inversion and resampling, so keep the input type.

**Randomness goes through `self.R`.** `RandomizableTransform` owns a `numpy.random.RandomState`
that `set_random_state` configures, and `randomize()` runs it on every call. Using the global
`np.random` bypasses the seed, and the transform is no longer reproducible.

**Invertibility has a contract.** Inherit `InvertibleTransform`, call `push_transform` in
`__call__` and `pop_transform` in `inverse`, and keep the extra information needed by the inverse
in `extra_info`. `extra_info` may only hold values that survive the data loader's collate
function, so `None` is not allowed. The stack itself lives in
`MetaTensor.applied_operations` and is unwound in last-in-first-out order.

**Lazy resampling is opt-in and deferred.** A `LazyTransform` describes its operation as an affine
matrix and lets the pipeline resample once, at the end. `requires_current_data` marks the
transforms that cannot be deferred, and `ApplyPending` forces the pending resampling to happen.
Turning `lazy` on changes when the pixels are computed, so a change here needs a test that pins
the output, not only the shape.

**A new transform is also a docs and export change.** Re-export it from
`monai/transforms/__init__.py`, add an `.. autoclass::` entry in `docs/source/transforms.rst`, and
add the module to `docs/source/transforms_idx.rst` when it is a new module.

**Tests mirror the family.** `tests/transforms/test_<name>.py` covers the array transform and
`tests/transforms/test_<name>d.py` the dictionary one; the inversion tests live in
`tests/transforms/inverse/`. A transform that is not covered there is not covered.
