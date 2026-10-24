# Data

## Key types

- `MetaTensor` (`meta_tensor.py`) is a `torch.Tensor` subclass carrying image
  metadata, affine information, and the record of applied transforms used by
  inverse transforms and lazy resampling. Code that silently converts it to a
  plain tensor (for example through `np.asarray` round trips) loses that
  information; preserve the type unless dropping it is the point.
- Datasets live in `dataset.py`: `Dataset`, `CacheDataset`,
  `SmartCacheDataset`, `PersistentDataset`, and friends. Readers and writers
  are registered in `image_reader.py` and `image_writer.py`.

## Things that are easy to get wrong

- `CacheDataset` caches the deterministic part of the pipeline. Its
  `copy_cache=True` default deep-copies cached items before the random
  transforms run; turning it off for speed reuses the cached tensors and can
  corrupt the cache for the next epoch.
- Optional formats (nibabel, pydicom, PIL, and others) are loaded through
  `monai.utils.optional_import` inside the readers. A reader must still
  import cleanly when its backend package is missing.
- `monai.data.DataLoader` extends the PyTorch loader with MONAI defaults;
  check it before reaching for `torch.utils.data.DataLoader` in MONAI code.
- Test images and other binary data do not belong in this repository. Tests
  fetch data through `tests.utils` helpers from the MONAI-extra-test-data
  releases; see `tests/AGENTS.md`.
