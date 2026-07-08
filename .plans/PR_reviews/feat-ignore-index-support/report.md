# PR Review Report — `feat-ignore-index-support`

## 1) PR Summary
Adds an `ignore_index` parameter to segmentation losses (DiceLoss, FocalLoss, TverskyLoss, AsymmetricUnifiedFocalLoss) and metrics (DiceMetric, MeanIoU, GeneralizedDiceScore, HausdorffDistanceMetric, SurfaceDiceMetric, SurfaceDistanceMetric, ConfusionMatrixMetric). A centralized `create_ignore_mask` helper in `monai/metrics/utils.py` generates spatial masks for label-encoded and one-hot targets, while `mask_loss_inputs` in `monai/losses/utils.py` applies them.

**Note:** No PR has been opened — this review is against branch `feat-ignore-index-support` vs `upstream/dev`.

## 2) Template Compliance
- [x] No PR opened yet — template compliance N/A (review is pre-submission)

### Notes
- No PR body to verify claims against.

## 3) Findings by Severity

### Critical
- None.

### Major

- **Title:** `DiceHelper.__call__` `first_ch` logic change alters per_component behavior
- **Severity:** Major
- **Evidence:** `monai/metrics/meandice.py:437` — line changed from `first_ch = 0 if self.include_background and not self.per_component else 1` to `first_ch = 0 if self.include_background else 1`.
- **Why it matters:** In `per_component=True` + `include_background=True` mode, the old code intentionally skipped channel 0 (background) and only computed Dice on the foreground channel. The new code includes channel 0, which is semantically wrong for per_component mode (the background channel has no connected components to analyze) and changes output values for existing users.
- **Suggested fix:** Restore the `and not self.per_component` condition: `first_ch = 0 if self.include_background and not self.per_component else 1`.

- **Title:** Cross-package internal import: losses importing from `monai.metrics.utils`
- **Severity:** Major
- **Evidence:** Four loss files import `create_ignore_mask` from `monai.metrics.utils`: `monai/losses/dice.py:26`, `monai/losses/focal_loss.py:22`, `monai/losses/tversky.py:21`, `monai/losses/unified_focal_loss.py:20`. Additionally, `monai/losses/utils.py:17` imports from the same metrics module.
- **Why it matters:** Losses and metrics are sibling packages. Having losses depend on metrics internals at the module level creates a soft import cycle (metrics already import from losses for `LossMetric`). This is architecturally fragile and against the principle that utility layers should not depend on their peers.
- **Suggested fix:** Move `create_ignore_mask` to a shared utility module (e.g., `monai/utils/` or a new `monai/losses/utils.py` copy) or accept the cycle but document it clearly.

- **Title:** `create_ignore_mask` not publicly exported from `monai.metrics` package
- **Severity:** Major
- **Evidence:** `monai/metrics/utils.py:49` adds `"create_ignore_mask"` to `__all__`, but `monai/metrics/__init__.py` does not import it (line 45 imports other utils but omits `create_ignore_mask`). The loss modules import it via the internal path `monai.metrics.utils.create_ignore_mask`, bypassing the package's public API.
- **Why it matters:** Users cannot `from monai.metrics import create_ignore_mask`. The function is effectively private yet used as a cross-package dependency. This inconsistency means external code that needs the same masking logic has no supported import path.
- **Suggested fix:** Add `create_ignore_mask` to the `monai/metrics/__init__.py` imports (line 45), or move it to a shared location and export from there.

### Minor

- **Title:** `mask_loss_inputs` docstring is too minimal
- **Severity:** Minor
- **Evidence:** `monai/losses/utils.py:76` — the docstring is a single line: `"""Apply ignore_index masking to loss inputs."""`
- **Why it matters:** This is a public utility function used by multiple loss classes. It should document its parameters, return type, and contract.
- **Suggested fix:** Add full Args/Returns docstring following Google style used throughout the codebase.

- **Title:** Duplicate `to_onehot_y=True` in test parameterization
- **Severity:** Minor
- **Evidence:** `tests/losses/test_ignore_index_losses.py:98` — `SENTINEL_ONEHOT_TEST_CASES` already includes `kwargs` that may contain `to_onehot_y`, but line 101 adds `to_onehot_y=True` again.
- **Why it matters:** This could cause subtle test failures if a future loss class rejects duplicate keyword arguments. Currently benign for `dict.update()` but fragile.
- **Suggested fix:** Remove the redundant explicit `to_onehot_y=True` from line 101 and rely on kwargs.

- **Title:** `AsymmetricUnifiedFocalLoss.forward` shape validation rewrite is complex
- **Severity:** Minor
- **Evidence:** `monai/losses/unified_focal_loss.py:250-286` — the shape validation logic has been extensively rewritten, adding ~30 lines of conditional checks for `to_onehot_y`, `ignore_index`, binary-to-2-channel conversion, and sentinel value handling.
- **Why it matters:** The original code had a simple `torch.max(y_true) != self.num_classes - 1` check. The new logic has many branching paths that are harder to reason about. A regression in the shape validation path could silently pass incorrect inputs.
- **Suggested fix:** Consider extracting the shape validation into a private helper method. Add test cases specifically for the new shape validation branches.

- **Title:** `get_surface_distance` type narrowing on seg_pred introduces dtype coupling
- **Severity:** Minor
- **Evidence:** `monai/metrics/utils.py:345-349` — new `if isinstance(seg_pred, torch.Tensor)` / `else` branch replaces a single generic `dis[seg_pred]` call.
- **Why it matters:** The old code relied on duck-typing (indexing worked for both torch and numpy). The new code bakes in a torch/numpy split. For cupy inputs this may break since they're not handled by the else branch.
- **Suggested fix:** Test with cupy inputs or add a cupy branch using `cupy.asarray(seg_pred).astype(bool)`.

## 4) Testing Assessment
- **Existing tests:** Two new test files (`test_ignore_index_losses.py`, `test_ignore_index_metrics.py`) cover ignore_index consistency, no-ignore behavior, class-index masking, and sentinel one-hot masking. Good coverage of the ignore_index feature paths.
- **Missing tests:** No tests for:
  - `DiceHelper` per_component mode with `ignore_index`
  - Shape validation edge cases in `AsymmetricUnifiedFocalLoss`
  - `get_edge_surface_distance` with `mask` and `warn_empty` parameters
  - `use_subvoxels=True` path with the new areas handling in `get_edge_surface_distance`
  - `compute_hausdorff_distance` with `ignore_index` matching a class index (NaN path)
- **Confidence:** Medium — the ignore_index core path is well tested, but the per_component regression and `get_edge_surface_distance` changes lack coverage.

## 5) Needs Author Clarification (if any)
- Was the removal of `and not self.per_component` from the `first_ch` assignment in `DiceHelper.__call__` intentional? If not, this is a regression.
- Is the losses → metrics import direction acceptable to maintainers, or should `create_ignore_mask` be extracted to a shared utility module?

## 6) Verdict
**Verdict:** Request Changes

The `first_ch` logic change in `DiceHelper.__call__` is likely a regression that silently alters Dice scores in per_component mode. The cross-package import direction and missing public export of `create_ignore_mask` should be resolved before merge.
