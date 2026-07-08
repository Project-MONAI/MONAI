# PR Review Comments (Copy/Paste Ready)

## Comment 1
- **File:** `tests/data/test_persistentdataset.py`
- **Line:** `216`
- **Severity:** Minor
- **Comment:** Leftover commented-out line from refactoring. The old `os.path.join` approach is replaced by `Path` usage on line 217, so this comment is dead code.
- **Suggested change:** Remove line 216.

## Comment 2
- **File:** `monai/data/dataset.py`
- **Line:** `387`
- **Severity:** Minor
- **Comment:** The local variable is named `data_item_md5` but the hash function now uses sha256 (changed in `utils.py`). This name is misleading — a future reader might assume md5 is still in use. Same issue on line 1621 for `CacheNTransDataset`.
- **Suggested change:** Rename `data_item_md5` to `data_item_hash` in both locations (lines 387-389 and 1621-1623).
