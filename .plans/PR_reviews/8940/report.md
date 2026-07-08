# PR Review Report — #8940

## 1) PR Summary
Allows `PersistentDataset` to cache `MetaTensor` objects with `weights_only=True` by leveraging MONAI's existing `torch.serialization.add_safe_globals([MetaTensor, ...])` registration. Switches cache-key hashing from md5 to sha256. Addresses GHSA-636w-j999-g7x5.

## 2) Template Compliance
- [x] Description matches implemented changes
- [x] Linked issue(s) are relevant — security advisory linked
- [x] Checklist claims match actual changes — new tests added, docstrings updated
- [ ] Type of change label is accurate — marked "Non-breaking" but see notes below

### Notes
- Marked "Non-breaking change" but the hash-algorithm change (md5 → sha256) invalidates all existing cache files. Old cache files won't be found (hash mismatch), forcing full recomputation and leaving orphaned .pt files on disk. Functionally the API is non-breaking, but users with large pre-built caches will experience an unexpected performance regression. Consider calling this out in a migration note or release changelog.

## 3) Findings by Severity

### Critical
- None.

### Major
- **Title:** Hash algorithm change invalidates all existing persistent caches
- **Severity:** Major
- **Evidence:** `monai/data/utils.py:1366-1383` (both `json_hashing` and `pickle_hashing`)
- **Why it matters:** Cache filenames are derived from hash output. After upgrading, none of the existing cache files will match the new sha256 hashes. All data will be recomputed, and old .pt files become orphaned on disk. For users with large cached datasets, this is a significant performance regression and disk-waste concern with no warning.
- **Suggested fix:** Document prominently in the PR description / changelog. Consider adding a one-time migration helper or a deprecation cycle (accept both hash formats for one release). At minimum, warn users to manually clear their cache directories after upgrading.

### Minor
- **Title:** Dead code comment in new test
- **Severity:** Minor
- **Evidence:** `tests/data/test_persistentdataset.py:216`
- **Why it matters:** The commented-out line `# cache_dir = os.path.join(os.path.join(tempdir, "cache"), "data")` is leftover from refactoring to `Path`. It's noise.
- **Suggested fix:** Remove the commented-out line.

- **Title:** Misleading variable name `data_item_md5` persists
- **Severity:** Minor
- **Evidence:** `monai/data/dataset.py:387-389`
- **Why it matters:** After the sha256 switch, the variable name `data_item_md5` is misleading. While pre-existing to this PR, the algorithm change makes this name actively confusing for future readers.
- **Suggested fix:** Rename to `data_item_hash` or similar. Same for the duplicate at line 1621.

## 4) Testing Assessment
- **Existing tests:** `test_track_meta_and_weights_only` updated — TEST_CASE_5 now expects `MetaTensor` instead of `ValueError`. Covers the new valid combination.
- **Missing tests:** No test for cache-key collision across hash algorithm change (would require a migration scenario). No test verifying old md5-named cache files are handled gracefully (they're silently ignored — is that the intended behavior?).
- **Confidence:** Medium — core logic (MetaTensor + weights_only) is well tested. Cache migration behavior is untested.

## 5) Needs Author Clarification (if any)
- Was the silent invalidation of all existing cache files intentional, or should there be a fallback/compatibility path? The PR description calls this "non-breaking" but the behavioral impact on cached datasets is material.

## 6) Verdict
**Verdict:** Approve with comments

The core change is sound: removing the artificial `track_meta=True` + `weights_only=True` restriction is correct since MetaTensor is registered as a safe global. Tests are thorough and cover both happy path and unsafe-rejection scenarios. The hash algorithm switch to sha256 is a security improvement. Main concern is the undocumented cache-invalidation impact — this should be communicated to users in release notes.
