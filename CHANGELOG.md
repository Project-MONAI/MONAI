# Changelog
All notable changes to MONAI are documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.6.0] - 2021-07-08
### Added
* 10 new transforms, a masked loss wrapper, and a `NetAdapter` for transfer learning
* APIs to load networks and pre-trained weights from Clara Train [Medical Model ARchives (MMARs)](https://docs.nvidia.com/clara/clara-train-sdk/pt/mmar.html)
* Base metric and cumulative metric APIs, 4 new regression metrics
* Initial CSV dataset support
* Decollating mini-batch as the default first postprocessing step, [Migrating your v0.5 code to v0.6](https://github.com/Project-MONAI/MONAI/wiki/v0.5-to-v0.6-migration-guide) wiki shows how to adapt the breaking changes
* Initial backward compatibility support via `monai.utils.deprecated`
* Attention-based vision modules and `UNETR` for segmentation
* Generic module loaders and Gaussian mixture models using the PyTorch JIT compilation
* Inverse of image patch sampling transforms
* Network block utilities `get_[norm, act, dropout, pool]_layer`
* `unpack_items` mode for `apply_transform` and `Compose`
* New event `INNER_ITERATION_STARTED` in the deepgrow interactive workflow
* `set_data` API for cache-based datasets to dynamically update the dataset content
* Fully compatible with PyTorch 1.9
* `--disttests` and `--min` options for `runtests.sh`
* Initial support of pre-merge tests with Nvidia Blossom system
### Changed
* Base Docker image upgraded to `nvcr.io/nvidia/pytorch:21.06-py3` from
  `nvcr.io/nvidia/pytorch:21.04-py3`
* Optionally depend on PyTorch-Ignite v0.4.5 instead of v0.4.4
* Unified the demo, tutorial, testing data to the project shared drive, and
  [`Project-MONAI/MONAI-extra-test-data`](https://github.com/Project-MONAI/MONAI-extra-test-data)
* Unified the terms: `post_transform` is renamed to `postprocessing`, `pre_transform` is renamed to `preprocessing`
* Unified the postprocessing transforms and event handlers to accept the "channel-first" data format
* `evenly_divisible_all_gather` and `string_list_all_gather` moved to `monai.utils.dist`
### Removed
* Support of 'batched' input for postprocessing transforms and event handlers
* `TorchVisionFullyConvModel`
* `set_visible_devices` utility function
* `SegmentationSaver` and `TransformsInverter` handlers
### Fixed
* Issue of handling big-endian image headers
* Multi-thread issue for non-random transforms in the cache-based datasets
* Persistent dataset issue when multiple processes sharing a non-exist cache location
* Typing issue with Numpy 1.21.0
* Loading checkpoint with both `model` and `optmizier` using `CheckpointLoader` when `strict_shape=False`
* `SplitChannel` has different behaviour depending on numpy/torch inputs
* Transform pickling issue caused by the Lambda functions
* Issue of filtering by name in `generate_param_groups`
* Inconsistencies in the return value types of `class_activation_maps`
* Various docstring typos
* Various usability enhancements in `monai.transforms`

## [0.5.3] - 2021-05-28
### Changed
* Project default branch renamed to `dev` from `master`
* Base Docker image upgraded to `nvcr.io/nvidia/pytorch:21.04-py3` from `nvcr.io/nvidia/pytorch:21.02-py3`
* Enhanced type checks for the `iteration_metric` handler
* Enhanced `PersistentDataset` to use `tempfile` during caching computation
* Enhanced various info/error messages
* Enhanced performance of `RandAffine`
* Enhanced performance of `SmartCacheDataset`
* Optionally requires `cucim` when the platform is `Linux`
* Default `device` of `TestTimeAugmentation` changed to `cpu`

### Fixed
* Download utilities now provide better default parameters
* Duplicated `key_transforms` in the patch-based transforms
* A multi-GPU issue in `ClassificationSaver`
* A default `meta_data` issue in `SpacingD`
* Dataset caching issue with the persistent data loader workers
* A memory issue in `permutohedral_cuda`
* Dictionary key issue in `CopyItemsd`
* `box_start` and `box_end` parameters for deepgrow `SpatialCropForegroundd`
* Tissue mask array transpose issue in `MaskedInferenceWSIDataset`
* Various type hint errors
* Various docstring typos

### Added
* Support of `to_tensor` and `device` arguments for `TransformInverter`
* Slicing options with SpatialCrop
* Class name alias for the networks for backward compatibility
* `k_divisible` option for CropForeground
* `map_items` option for `Compose`
* Warnings of `inf` and `nan` for surface distance computation
* A `print_log` flag to the image savers
* Basic testing pipelines for Python 3.9

## [0.5.0] - 2021-04-09
### Added
* Overview document for [feature highlights in v0.5.0](https://github.com/Project-MONAI/MONAI/blob/master/docs/source/highlights.md)
* Invertible spatial transforms
  * `InvertibleTransform` base APIs
  * Batch inverse and decollating APIs
  * Inverse of `Compose`
  * Batch inverse event handling
  * Test-time augmentation as an application
* Initial support of learning-based image registration:
  * Bending energy, LNCC, and global mutual information loss
  * Fully convolutional architectures
  * Dense displacement field, dense velocity field computation
  * Warping with high-order interpolation with C++/CUDA implementations
* Deepgrow modules for interactive segmentation:
  * Workflows with simulations of clicks
  * Distance-based transforms for guidance signals
* Digital pathology support:
  * Efficient whole slide imaging IO and sampling with Nvidia cuCIM and SmartCache
  * FROC measurements for lesion
  * Probabilistic post-processing for lesion detection
  * TorchVision classification model adaptor for fully convolutional analysis
* 12 new transforms, grid patch dataset, `ThreadDataLoader`, EfficientNets B0-B7
* 4 iteration events for the engine for finer control of workflows
* New C++/CUDA extensions:
  * Conditional random field
  * Fast bilateral filtering using the permutohedral lattice
* Metrics summary reporting and saving APIs
* DiceCELoss, DiceFocalLoss, a multi-scale wrapper for segmentation loss computation
* Data loading utilities：
  * `decollate_batch`
  * `PadListDataCollate` with inverse support
* Support of slicing syntax for `Dataset`
* Initial Torchscript support for the loss modules
* Learning rate finder
* Allow for missing keys in the dictionary-based transforms
* Support of checkpoint loading for transfer learning
* Various summary and plotting utilities for Jupyter notebooks
* Contributor Covenant Code of Conduct
* Major CI/CD enhancements covering the tutorial repository
* Fully compatible with PyTorch 1.8
* Initial nightly CI/CD pipelines using Nvidia Blossom Infrastructure

### Changed
* Enhanced `list_data_collate` error handling
* Unified iteration metric APIs
* `densenet*` extensions are renamed to `DenseNet*`
* `se_res*` network extensions are renamed to `SERes*`
* Transform base APIs are rearranged into `compose`, `inverse`, and `transform`
* `_do_transform` flag for the random augmentations is unified via `RandomizableTransform`
* Decoupled post-processing steps, e.g. `softmax`, `to_onehot_y`, from the metrics computations
* Moved the distributed samplers to `monai.data.samplers` from `monai.data.utils`
* Engine's data loaders now accept generic iterables as input
* Workflows now accept additional custom events and state properties
* Various type hints according to Numpy 1.20
* Refactored testing utility `runtests.sh` to have `--unittest` and `--net` (integration tests) options
* Base Docker image upgraded to `nvcr.io/nvidia/pytorch:21.02-py3` from `nvcr.io/nvidia/pytorch:20.10-py3`
* Docker images are now built with self-hosted environments
* Primary contact email updated to `monai.contact@gmail.com`
* Now using GitHub Discussions as the primary communication forum

### Removed
* Compatibility tests for PyTorch 1.5.x
* Format specific loaders, e.g. `LoadNifti`, `NiftiDataset`
* Assert statements from non-test files
* `from module import *` statements, addressed flake8 F403

### Fixed
* Uses American English spelling for code, as per PyTorch
* Code coverage now takes multiprocessing runs into account
* SmartCache with initial shuffling
* `ConvertToMultiChannelBasedOnBratsClasses` now supports channel-first inputs
* Checkpoint handler to save with non-root permissions
* Fixed an issue for exiting the distributed unit tests
* Unified `DynUNet` to have single tensor output w/o deep supervision
* `SegmentationSaver` now supports user-specified data types and a `squeeze_end_dims` flag
* Fixed `*Saver` event handlers output filenames with a `data_root_dir` option
* Load image functions now ensure little-endian
* Fixed the test runner to support regex-based test case matching
* Usability issues in the event handlers

## [0.4.0] - 2020-12-15
### Added
* Overview document for [feature highlights in v0.4.0](https://github.com/Project-MONAI/MONAI/blob/master/docs/source/highlights.md)
* Torchscript support for the net modules
* New networks and layers:
  * Discrete Gaussian kernels
  * Hilbert transform and envelope detection
  * Swish and mish activation
  * Acti-norm-dropout block
  * Upsampling layer
  * Autoencoder, Variational autoencoder
  * FCNet
* Support of initialisation from pretrained weights for densenet, senet, multichannel AHNet
* Layer-wise learning rate API
* New model metrics and event handlers based on occlusion sensitivity, confusion matrix, surface distance
* CAM/GradCAM/GradCAM++
* File format-agnostic image loader APIs with Nibabel, ITK readers
* Enhancements for dataset partition, cross-validation APIs
* New data APIs:
  * LMDB-based caching dataset
  * Cache-N-transforms dataset
  * Iterable dataset
  * Patch dataset
* Weekly PyPI release
* Fully compatible with PyTorch 1.7
* CI/CD enhancements:
  * Skipping, speed up, fail fast, timed, quick tests
  * Distributed training tests
  * Performance profiling utilities
* New tutorials and demos:
  * Autoencoder, VAE tutorial
  * Cross-validation demo
  * Model interpretability tutorial
  * COVID-19 Lung CT segmentation challenge open-source baseline
  * Threadbuffer demo
  * Dataset partitioning tutorial
  * Layer-wise learning rate demo
  * [MONAI Bootcamp 2020](https://github.com/Project-MONAI/MONAIBootcamp2020)

### Changed
* Base Docker image upgraded to `nvcr.io/nvidia/pytorch:20.10-py3` from `nvcr.io/nvidia/pytorch:20.08-py3`

#### Backwards Incompatible Changes
* `monai.apps.CVDecathlonDataset` is extended to a generic `monai.apps.CrossValidation` with an `dataset_cls` option
* Cache dataset now requires a `monai.transforms.Compose` instance as the transform argument
* Model checkpoint file name extensions changed from `.pth` to `.pt`
* Readers' `get_spatial_shape` returns a numpy array instead of list
* Decoupled postprocessing steps such as `sigmoid`, `to_onehot_y`, `mutually_exclusive`, `logit_thresh` from metrics and event handlers,
the postprocessing steps should be used before calling the metrics methods
* `ConfusionMatrixMetric` and `DiceMetric` computation now returns an additional `not_nans` flag to indicate valid results
* `UpSample` optional `mode` now supports `"deconv"`, `"nontrainable"`, `"pixelshuffle"`; `interp_mode` is only used when `mode` is `"nontrainable"`
* `SegResNet` optional `upsample_mode` now supports `"deconv"`, `"nontrainable"`, `"pixelshuffle"`
* `monai.transforms.Compose` class inherits `monai.transforms.Transform`
* In `Rotate`, `Rotated`, `RandRotate`, `RandRotated`  transforms, the `angle` related parameters are interpreted as angles in radians instead of degrees.
* `SplitChannel` and `SplitChanneld` moved from `transforms.post` to `transforms.utility`

### Removed
* Support of PyTorch 1.4

### Fixed
* Enhanced loss functions for stability and flexibility
* Sliding window inference memory and device issues
* Revised transforms:
  * Normalize intensity datatype and normalizer types
  * Padding modes for zoom
  * Crop returns coordinates
  * Select items transform
  * Weighted patch sampling
  * Option to keep aspect ratio for zoom
* Various CI/CD issues

## [0.3.0] - 2020-10-02
### Added
* Overview document for [feature highlights in v0.3.0](https://github.com/Project-MONAI/MONAI/blob/master/docs/source/highlights.md)
* Automatic mixed precision support
* Multi-node, multi-GPU data parallel model training support
* 3 new evaluation metric functions
* 11 new network layers and blocks
* 6 new network architectures
* 14 new transforms, including an I/O adaptor
* Cross validation module for `DecathlonDataset`
* Smart Cache module in dataset
* `monai.optimizers` module
* `monai.csrc` module
* Experimental feature of ImageReader using ITK, Nibabel, Numpy, Pillow (PIL Fork)
* Experimental feature of differentiable image resampling in C++/CUDA
* Ensemble evaluator module
* GAN trainer module
* Initial cross-platform CI environment for C++/CUDA code
* Code style enforcement now includes isort and clang-format
* Progress bar with tqdm

### Changed
* Now fully compatible with PyTorch 1.6
* Base Docker image upgraded to `nvcr.io/nvidia/pytorch:20.08-py3` from `nvcr.io/nvidia/pytorch:20.03-py3`
* Code contributions now require signing off on the [Developer Certificate of Origin (DCO)](https://developercertificate.org/)
* Major work in type hinting finished
* Remote datasets migrated to [Open Data on AWS](https://registry.opendata.aws/)
* Optionally depend on PyTorch-Ignite v0.4.2 instead of v0.3.0
* Optionally depend on torchvision, ITK
* Enhanced CI tests with 8 new testing environments
### Removed
* `MONAI/examples` folder (relocated into [`Project-MONAI/tutorials`](https://github.com/Project-MONAI/tutorials))
* `MONAI/research` folder (relocated to [`Project-MONAI/research-contributions`](https://github.com/Project-MONAI/research-contributions))
### Fixed
* `dense_patch_slices` incorrect indexing
* Data type issue in `GeneralizedWassersteinDiceLoss`
* `ZipDataset` return value inconsistencies
* `sliding_window_inference` indexing and `device` issues
* importing monai modules may cause namespace pollution
* Random data splits issue in `DecathlonDataset`
* Issue of randomising a `Compose` transform
* Various issues in function type hints
* Typos in docstring and documentation
* `PersistentDataset` issue with existing file folder
* Filename issue in the output writers

## [0.2.0] - 2020-07-02
### Added
* Overview document for [feature highlights in v0.2.0](https://github.com/Project-MONAI/MONAI/blob/master/docs/source/highlights.md)
* Type hints and static type analysis support
* `MONAI/research` folder
* `monai.engine.workflow` APIs for supervised training
* `monai.inferers` APIs for validation and inference
* 7 new tutorials and examples
* 3 new loss functions
* 4 new event handlers
* 8 new layers, blocks, and networks
* 12 new transforms, including post-processing transforms
* `monai.apps.datasets` APIs, including `MedNISTDataset` and `DecathlonDataset`
* Persistent caching, `ZipDataset`, and `ArrayDataset` in `monai.data`
* Cross-platform CI tests supporting multiple Python versions
* Optional import mechanism
* Experimental features for third-party transforms integration
### Changed
> For more details please visit [the project wiki](https://github.com/Project-MONAI/MONAI/wiki/Notable-changes-between-0.1.0-and-0.2.0)
* Core modules now require numpy >= 1.17
* Categorized `monai.transforms` modules into crop and pad, intensity, IO, post-processing, spatial, and utility.
* Most transforms are now implemented with PyTorch native APIs
* Code style enforcement and automated formatting workflows now use autopep8 and black
* Base Docker image upgraded to `nvcr.io/nvidia/pytorch:20.03-py3` from `nvcr.io/nvidia/pytorch:19.10-py3`
* Enhanced local testing tools
* Documentation website domain changed to https://docs.monai.io
### Removed
* Support of Python < 3.6
* Automatic installation of optional dependencies including pytorch-ignite, nibabel, tensorboard, pillow, scipy, scikit-image
### Fixed
* Various issues in type and argument names consistency
* Various issues in docstring and documentation site
* Various issues in unit and integration tests
* Various issues in examples and notebooks

## [0.1.0] - 2020-04-17
### Added
* Public alpha source code release under the Apache 2.0 license ([highlights](https://github.com/Project-MONAI/MONAI/blob/0.1.0/docs/source/highlights.md))
* Various tutorials and examples
  - Medical image classification and segmentation workflows
  - Spacing/orientation-aware preprocessing with CPU/GPU and caching
  - Flexible workflows with PyTorch Ignite and Lightning
* Various GitHub Actions
  - CI/CD pipelines via self-hosted runners
  - Documentation publishing via readthedocs.org
  - PyPI package publishing
* Contributing guidelines
* A project logo and badges

[highlights]: https://github.com/Project-MONAI/MONAI/blob/master/docs/source/highlights.md

[Unreleased]: https://github.com/Project-MONAI/MONAI/compare/0.6.0...HEAD
[0.6.0]: https://github.com/Project-MONAI/MONAI/compare/0.5.3...0.6.0
[0.5.3]: https://github.com/Project-MONAI/MONAI/compare/0.5.0...0.5.3
[0.5.0]: https://github.com/Project-MONAI/MONAI/compare/0.4.0...0.5.0
[0.4.0]: https://github.com/Project-MONAI/MONAI/compare/0.3.0...0.4.0
[0.3.0]: https://github.com/Project-MONAI/MONAI/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/Project-MONAI/MONAI/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/Project-MONAI/MONAI/commits/0.1.0
