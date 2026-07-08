# PR Review Comments (Copy/Paste Ready)

## Comment 1
- **File:** `monai/metrics/meandice.py`
- **Line:** `437`
- **Severity:** Major
- **Comment:** The `and not self.per_component` condition was removed from the `first_ch` assignment. In `per_component=True` + `include_background=True` mode, the old code correctly skipped channel 0 (background has no connected components to analyze). The new code includes it, which changes Dice scores for per_component users.
- **Suggested change:** Restore the condition: `first_ch = 0 if self.include_background and not self.per_component else 1`

## Comment 2
- **File:** `monai/metrics/__init__.py`
- **Line:** `45`
- **Severity:** Major
- **Comment:** `create_ignore_mask` is added to `__all__` in `monai/metrics/utils.py` but not imported here. Since losses modules import it from `monai.metrics.utils`, it should be publicly available. External code that needs the same masking logic has no supported import path.
- **Suggested change:** Add `create_ignore_mask` to the import from `.utils` on this line.

## Comment 3
- **File:** `monai/losses/utils.py`
- **Line:** `76`
- **Severity:** Minor
- **Comment:** The docstring `"""Apply ignore_index masking to loss inputs."""` is too minimal for a utility function used by four loss classes. Please document the parameters, return type, and contract (e.g., what happens when mask and ignore_index are both None vs one set).
- **Suggested change:** Add Google-style Args/Returns docstring.

## Comment 4
- **File:** `tests/losses/test_ignore_index_losses.py`
- **Line:** `101`
- **Severity:** Minor
- **Comment:** `to_onehot_y=True` is passed explicitly here, but some test case kwargs in `SENTINEL_ONEHOT_TEST_CASES` already include it. This could cause a `TypeError` if a loss class rejects duplicate keyword arguments in the future.
- **Suggested change:** Remove the explicit `to_onehot_y=True` and rely on the kwargs dict, or ensure kwargs never carry `to_onehot_y`.

## Comment 5
- **File:** `monai/losses/unified_focal_loss.py`
- **Line:** `250`
- **Severity:** Minor
- **Comment:** The shape validation logic in `AsymmetricUnifiedFocalLoss.forward` grew from ~5 lines to ~30+ lines of conditionals. This is now significantly harder to audit for correctness. Consider extracting the validation into a private `_validate_and_prepare_inputs` helper to keep `forward` focused on the loss computation.
- **Suggested change:** Extract shape validation to a private method; add focused tests for the new branches (binary-to-2-channel, sentinel ignore_index before one_hot, mismatch scenarios).
