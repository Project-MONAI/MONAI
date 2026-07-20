
# What's new in 1.6.0 🎉🎉

- New losses: AUC-Margin Loss, Matthews Correlation Coefficient (MCC) Loss
  * AUC-Margin Loss: `AUCMarginLoss` directly optimizes the AUROC metric via a margin-based surrogate loss, enabling training workflows that target ranking-based performance rather than calibration-based objectives.
  * Matthews Correlation Coefficient (MCC) Loss: `MCCLoss` provides a differentiable loss based on the Matthews Correlation Coefficient, which accounts for all four confusion matrix categories and is particularly robust for imbalanced segmentation tasks.
- New metric: `EmbeddingCollapseMetric` detects representational collapse in learned embedding spaces, useful for self-supervised and contrastive learning workflows in medical imaging to monitor embedding quality during training.
- Whole Slide Image (WSI) reader now supports retrieval at a specified microns-per-pixel (MPP) resolution. This simplifies multi-scanner workflows where consistent physical-space resolution is required regardless of scanner magnification levels.
- Nested dot-notation key access in `ConfigParser`.
- Auto3DSeg algo serialization migrated from pickle to JSON for improved security and portability.
- Global coordinates support in spatial crop transforms. These now support global coordinate mode, allowing crops to be specified in world/global coordinates rather than local image indices, improving interoperability with physical-space annotations.
- `SoftclDiceLoss` and `SoftDiceclDiceLoss` enhanced with `DiceLoss`-compatible API
- Variable expansion hardening has been added to the nnUNet app to eliminate code injection attacks when composing shell command lines, addressing concerns in [GHSA-rghg-q7wp-9767](https://github.com/Project-MONAI/MONAI/security/advisories/GHSA-rghg-q7wp-9767).
- `NumpyReader` has been updated with an `allow_pickle` boolean argument to enable/disable pickle loading from `.npy/.npz` files. This was previously hard-coded to be enabled, but is now defined by this argument and disabled by default. This addresses [GHSA-qxq5-qhx6-94qw](https://github.com/Project-MONAI/MONAI/security/advisories/GHSA-qxq5-qhx6-94qw).


MONAI now tests for Python 3.10 onwards, having dropped version 3.9 which is now out of support. PyTorch 2.8 onwards is now supported only, older versions will likely continue to function.

## Nested Dot-Notation Access in ConfigParser

`ConfigParser` now supports nested dot-notation key access, making it easier to read and override deeply nested configuration values programmatically.

For example, accessing a value from the parser with `parser["network_def.in_channels"]` can instead be `parser.network_def.in_channels`. This feature supports indexing and assignment, eg. `parser.network_def.in_channels[0] = 4` or `parser.A.B["C"] = 99`.

## Auto3DSeg: JSON-Based Algo Serialization

Auto3DSeg algorithm objects are now serialized using JSON instead of pickle. This removes a class of security risks associated with pickle deserialization and improves cross-environment portability of saved algorithm states. Using pickle for serialization can be re-enabled by setting the environment variable `MONAI_ALLOW_PICKLE` to `1` or the equivalent true value.

This was implemented to address [GHSA-qxq5-qhx6-94qw](https://github.com/Project-MONAI/MONAI/security/advisories/GHSA-qxq5-qhx6-94qw).

## SoftclDiceLoss / SoftDiceclDiceLoss API Alignment

`SoftclDiceLoss` and `SoftDiceclDiceLoss` now accept the same arguments as `DiceLoss`, including `reduction`, `smooth_nr`, `smooth_dr`, and `batch` parameters, enabling drop-in use alongside the standard Dice loss in existing pipelines.

## Minor Changes

- `DiceMetric` and `DiceHelper` accept additional parameters for finer control of reduction behavior
- `ExtractDataKeyFromMetaKeyd` now works with `MetaTensor` inputs
- `ConvertToMultiChannelBasedOnBratsClasses` supports configurable GD-enhancing tumor label
- TorchScript compatibility: replaced `Tensor | None` union syntax with `Optional[Tensor]` across network modules
- `CrossAttentionBlock` is now only instantiated when `with_cross_attention=True`, reducing memory overhead
- `GlobalMutualInformationLoss` bin centers and `LocalNormalizedCrossCorrelationLoss` kernels registered as buffers for correct device handling (`#8869`, `#8818`)
- `NibabelReader` avoids eager C-order memory copies, reducing peak RAM usage for large NIfTI files
- Fixed `align_corners` mismatch in `AffineTransform`
- Fixed nested `Compose` `map_items` behaviour in forward and inverse paths
- Fixed anchor centering on grid cells in detection
- Fixed multi-axis shear transform matrix composition
- Fixed `JukeboxLoss` swapped `input_amplitude`/`target_amplitude` arguments
- Fixed memory leak in `optional_import` traceback handling
- Fixed `RandSimulateLowResolution` to use `F.interpolate` instead of `set_track_meta`
- Fixed GPU memory leak when checking image/label device in engine utilities
- Fixed `AutoencoderKL` `proj_attn` → `out_proj` key remapping in `load_old_state_dict`
- Fixed incorrect `truncated` parameter in `make_gaussian_kernel` affecting `LocalNormalizedCrossCorrelationLoss`
- Fixed `compute_shape_offset` non-tuple indexing for PyTorch ≥ 2.9
- Auto3DSeg: fixed incorrect device resolution in analyzer and precomputed crop handling
- Replaced `np.random.*` global calls with `np.random.RandomState` instances for reproducibility
- Replaced `BaseException` with `Exception` across the codebase
