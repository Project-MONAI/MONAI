# Changelog
All notable changes to MONAI are documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [1.3.2] - 2024-06-25
### Fixed
#### misc.
* Updated Numpy version constraint to < 2.0 (#7859)

## [1.3.1] - 2024-05-17
### Added
* Support for `by_measure` argument in `RemoveSmallObjects` (#7137)
* Support for `pretrained` flag in `ResNet` (#7095)
* Support for uploading and downloading bundles to and from the Hugging Face Hub (#6454)
* Added weight parameter in DiceLoss to apply weight to voxels of each class (#7158)
* Support for returning dice for each class in `DiceMetric` (#7163)
* Introduced `ComponentStore` for storage purposes (#7159)
* Added utilities used in MONAI Generative (#7134)
* Enabled Python 3.11 support for `convert_to_torchscript` and `convert_to_onnx` (#7182)
* Support for MLflow in `AutoRunner` (#7176)
* `fname_regex` option in PydicomReader (#7181)
* Allowed setting AutoRunner parameters from config (#7175)
* `VoxelMorphUNet` and `VoxelMorph` (#7178)
* Enabled `cache` option in `GridPatchDataset` (#7180)
* Introduced `class_labels` option in `write_metrics_reports` for improved readability (#7249)
* `DiffusionLoss` for image registration task (#7272)
* Supported specifying `filename` in `Saveimage` (#7318)
* Compile support in `SupervisedTrainer` and `SupervisedEvaluator` (#7375)
* `mlflow_experiment_name` support in `Auto3DSeg` (#7442)
* Arm support (#7500)
* `BarlowTwinsLoss` for representation learning (#7530)
* `SURELoss` and `ConjugateGradient` for diffusion models (#7308)
* Support for `CutMix`, `CutOut`, and `MixUp` augmentation techniques (#7198)
* `meta_file` and `logging_file` options to `BundleWorkflow` (#7549)
* `properties_path` option to `BundleWorkflow` for customized properties (#7542)
* Support for both soft and hard clipping in `ClipIntensityPercentiles` (#7535)
* Support for not saving artifacts in `MLFlowHandler` (#7604)
* Support for multi-channel images in `PerceptualLoss` (#7568)
* Added ResNet backbone for `FlexibleUNet` (#7571)
* Introduced `dim_head` option in `SABlock` to set dimensions for each head (#7664)
* Direct links to github source code to docs (#7738, #7779)
#### misc.
* Refactored `list_data_collate` and `collate_meta_tensor` to utilize the latest PyTorch API (#7165)
* Added __str__ method in `Metric` base class (#7487)
* Made enhancements for testing files (#7662, #7670, #7663, #7671, #7672)
* Improved documentation for bundles (#7116)
### Fixed
#### transforms
* Addressed issue where lazy mode was ignored in `SpatialPadd` (#7316)
* Tracked applied operations in `ImageFilter` (#7395)
* Warnings are now given only if missing class is not set to 0 in `generate_label_classes_crop_centers` (#7602)
* Input is now always converted to C-order in `distance_transform_edt` to ensure consistent behavior (#7675)
#### data
* Modified .npz file behavior to use keys in `NumpyReader` (#7148)
* Handled corrupted cached files in `PersistentDataset` (#7244)
* Corrected affine update in `NrrdReader` (#7415)
#### metrics and losses
* Addressed precision issue in `get_confusion_matrix` (#7187)
* Harmonized and clarified documentation and tests for dice losses variants (#7587)
#### networks
* Removed hard-coded `spatial_dims` in `SwinTransformer` (#7302)
* Fixed learnable `position_embeddings` in `PatchEmbeddingBlock` (#7564, #7605)
* Removed `memory_pool_limit` in TRT config (#7647)
* Propagated `kernel_size` to `ConvBlocks` within `AttentionUnet` (#7734)
* Addressed hard-coded activation layer in `ResNet` (#7749)
#### bundle
* Resolved bundle download issue (#7280)
* Updated `bundle_root` directory for `NNIGen` (#7586)
* Checked for `num_fold` and failed early if incorrect (#7634)
* Enhanced logging logic in `ConfigWorkflow` (#7745)
#### misc.
* Enabled chaining in `Auto3DSeg` CLI (#7168)
* Addressed useless error message in `nnUNetV2Runner` (#7217)
* Resolved typing and deprecation issues in Mypy (#7231)
* Quoted `$PY_EXE` variable to handle Python path that contains spaces in Bash (#7268)
* Improved documentation, code examples, and warning messages in various modules (#7234, #7213, #7271, #7326, #7569, #7584)
* Fixed typos in various modules (#7321, #7322, #7458, #7595, #7612)
* Enhanced docstrings in various modules (#7245, #7381, #7746)
* Handled error when data is on CPU in `DataAnalyzer` (#7310)
* Updated version requirements for third-party packages (#7343, #7344, #7384, #7448, #7659, #7704, #7744, #7742, #7780)
* Addressed incorrect slice compute in `ImageStats` (#7374)
* Avoided editing a loop's mutable iterable to address B308 (#7397)
* Fixed issue with `CUDA_VISIBLE_DEVICES` setting being ignored (#7408, #7581)
* Avoided changing Python version in CICD (#7424)
* Renamed partial to callable in instantiate mode (#7413)
* Imported AttributeError for Python 3.12 compatibility (#7482)
* Updated `nnUNetV2Runner` to support nnunetv2 2.2 (#7483)
* Used uint8 instead of int8 in `LabelStats` (#7489)
* Utilized subprocess for nnUNet training (#7576)
* Addressed deprecated warning in ruff (#7625)
* Fixed downloading failure on FIPS machine (#7698)
* Updated `torch_tensorrt` compile parameters to avoid warning (#7714)
* Restrict `Auto3DSeg` fold input based on datalist (#7778)
### Changed
* Base Docker image upgraded to `nvcr.io/nvidia/pytorch:24.03-py3` from `nvcr.io/nvidia/pytorch:23.08-py3`
### Removed
* Removed unrecommended star-arg unpacking after a keyword argument, addressed B026 (#7262)
* Skipped old PyTorch version test for `SwinUNETR` (#7266)
* Dropped docker build workflow and migrated to Nvidia Blossom system (#7450)
* Dropped Python 3.8 test on quick-py3 workflow (#7719)

## [1.3.0] - 2023-10-12
### Added
* Intensity transforms `ScaleIntensityFixedMean` and `RandScaleIntensityFixedMean` (#6542)
* `UltrasoundConfidenceMapTransform` used for computing confidence map from an ultrasound image (#6709)
* `channel_wise` support in `RandScaleIntensity` and `RandShiftIntensity` (#6793, #7025)
* `RandSimulateLowResolution` and `RandSimulateLowResolutiond` (#6806)
* `SignalFillEmptyd` (#7011)
* Euclidean distance transform `DistanceTransformEDT` with GPU support (#6981)
* Port loss and metrics from `monai-generative` (#6729, #6836)
* Support `invert_image` and `retain_stats` in `AdjustContrast` and `RandAdjustContrast` (#6542)
* New network `DAF3D` and `Quicknat` (#6306)
* Support `sincos` position embedding (#6986)
* `ZarrAvgMerger` used for patch inference (#6633)
* Dataset tracking support to `MLFlowHandler` (#6616)
* Considering spacing and subvoxel borders in `SurfaceDiceMetric` (#6681)
* CUCIM support for surface-related metrics (#7008)
* `loss_fn` support in `IgniteMetric` and renamed it to `IgniteMetricHandler` (#6695)
* `CallableEventWithFilter` and `Events` options for `trigger_event` in `GarbageCollector` (#6663)
* Support random sorting option to `GridPatch`, `RandGridPatch`, `GridPatchd` and `RandGridPatchd` (#6701)
* Support multi-threaded batch sampling in `PatchInferer` (#6139)
* `SoftclDiceLoss` and `SoftDiceclDiceLoss` (#6763)
* `HausdorffDTLoss` and `LogHausdorffDTLoss` (#6994)
* Documentation for `TensorFloat-32` (#6770)
* Docstring format guide (#6780)
* `GDSDataset` support for GDS (#6778)
* PyTorch backend support for `MapLabelValue` (#6872)
* `filter_func` in `copy_model_state` to filter the weights to be loaded  and `filter_swinunetr` (#6917)
* `stats_sender` to `MonaiAlgo` for FL stats (#6984)
* `freeze_layers` to help freeze specific layers (#6970)
#### misc.
* Refactor multi-node running command used in `Auto3DSeg` into dedicated functions (#6623)
* Support str type annotation to `device` in `ToTensorD` (#6737)
* Improve logging message and file name extenstion in `DataAnalyzer` for `Auto3DSeg` (#6758)
* Set `data_range` as a property in `SSIMLoss` (#6788)
* Unify environment variable access (#7084)
* `end_lr` support in `WarmupCosineSchedule` (#6662)
* Add `ClearML` as optional dependency (#6827)
* `yandex.disk` support in `download_url` (#6667)
* Improve config expression error message (#6977)
### Fixed
#### transforms
* Make `convert_box_to_mask` throw errors when box size larger than the image (#6637)
* Fix lazy mode in `RandAffine` (#6774)
* Raise `ValueError` when `map_items` is bool in `Compose` (#6882)
* Improve performance for `NormalizeIntensity` (#6887)
* Fix mismatched shape in `Spacing` (#6912)
* Avoid FutureWarning in `CropForeground` (#6934)
* Fix `Lazy=True` ignored when using `Dataset` call (#6975)
* Shape check for arbitrary types for DataStats (#7082)
#### data
* Fix wrong spacing checking logic in `PydicomReader` and broken link in `ITKReader` (#6660)
* Fix boolean indexing of batched `MetaTensor` (#6781)
* Raise warning when multiprocessing in `DataLoader` (#6830)
* Remove `shuffle` in `DistributedWeightedRandomSampler` (#6886)
* Fix missing `SegmentDescription` in `PydicomReader` (#6937)
* Fix reading dicom series error in `ITKReader` (#6943)
* Fix KeyError in `PydicomReader` (#6946)
* Update `metatensor_to_itk_image` to accept RAS `MetaTensor` and update default 'space' in `NrrdReader` to `SpaceKeys.LPS` (#7000)
* Collate common meta dictionary keys (#7054)
#### metrics and losses
* Fixed bug in `GeneralizedDiceLoss` when `batch=True` (#6775)
* Support for `BCEWithLogitsLoss` in `DiceCELoss` (#6924)
* Support for `weight` in Dice and related losses (#7098)
#### networks
* Use `np.prod` instead of `np.product` (#6639)
* Fix dimension issue in `MBConvBlock` (#6672)
* Fix hard-coded `up_kernel_size` in `ViTAutoEnc` (#6735)
* Remove hard-coded `bias_downsample` in `resnet` (#6848)
* Fix unused `kernel_size` in `ResBlock` (#6999)
* Allow for defining reference grid on non-integer coordinates (#7032)
* Padding option for autoencoder (#7068)
* Lower peak memory usage for SegResNetDS (#7066)
#### bundle
* Set `train_dataset_data` and `dataset_data` to unrequired in BundleProperty (#6607)
* Set `None` to properties that do not have `REF_ID` (#6607)
* Fix `AttributeError` for default value in `get_parsed_content` for `ConfigParser` (#6756)
* Update `monai.bundle.scripts` to support NGC hosting (#6828, #6997)
* Add `MetaProperties` (#6835)
* Add `create_workflow` and update `load` function (#6835)
* Add bundle root directory to Python search directories automatically (#6910)
* Generate properties for bundle docs automatically (#6918)
* Move `download_large_files` from model zoo to core (#6958)
* Bundle syntax `#` as alias of `::` (#6955)
* Fix bundle download naming issue (#6969, #6963)
* Simplify the usage of `ckpt_export` (#6965)
* `update_kwargs` in `monai.bundle.script` for merging multiple configs (#7109)
#### engines and handlers
* Added int options for `iteration_log` and `epoch_log` in `TensorBoardStatsHandler` (#7027)
* Support to run validator at training start (#7108)
#### misc.
* Fix device fallback error in `DataAnalyzer` (#6658)
* Add int check for  `current_mode` in `convert_applied_interp_mode` (#6719)
* Consistent type in `convert_to_contiguous` (#6849)
* Label `argmax` in `DataAnalyzer` when retry on CPU (#6852)
* Fix `DataAnalyzer` with `histogram_only=True` (#6874)
* Fix `AttributeError` in `RankFilter` in single GPU environment (#6895)
* Remove the default warning on `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` and add debug print info (#6909)
* Hide user information in `print_config` (#6913, #6922)
* Optionally pass coordinates to predictor during sliding window (#6795)
* Proper ensembling when trained with a sigmoid in `AutoRunner` (#6588)
* Fixed `test_retinanet` by increasing absolute differences (#6615)
* Add type check to avoid comparing a np.array with a string in `_check_kwargs_are_present` (#6624)
* Fix md5 hashing with FIPS mode (#6635)
* Capture failures from Auto3DSeg related subprocess calls (#6596)
* Code formatting tool for user-specified directory (#7106)
* Various docstring fixes
### Changed
* Base Docker image upgraded to `nvcr.io/nvidia/pytorch:23.08-py3` from `nvcr.io/nvidia/pytorch:23.03-py3`
### Deprecated
* `allow_smaller=True`; `allow_smaller=False` will be the new default in `CropForeground` and `generate_spatial_bounding_box` (#6736)
* `dropout_prob` in `VNet` in favor of `dropout_prob_down` and `dropout_prob_up` (#6768)
* `workflow` in `BundleWorkflow` in favor of `workflow_type`(#6768)
* `pos_embed` in `PatchEmbeddingBlock` in favor of `proj_type`(#6986)
* `net_name` and `net_kwargs` in `download` in favor of `model`(#7016)
* `img_size` parameter in SwinUNETR (#7093)
### Removed
* `pad_val`, `stride`, `per_channel` and `upsampler` in `OcclusionSensitivity` (#6642)
* `compute_meaniou` (#7019)
* `AsChannelFirst`, `AddChannel`and `SplitChannel` (#7019)
* `create_multigpu_supervised_trainer` and `create_multigpu_supervised_evaluator` (#7019)
* `runner_id` in `run` (#7019)
* `data_src_cfg_filename` in `AlgoEnsembleBuilder` (#7019)
* `get_validation_stats` in `Evaluator` and `get_train_stats` in `Trainer` (#7019)
* `epoch_interval` and `iteration_interval` in `TensorBoardStatsHandler` (#7019)
* some self-hosted test (#7041)

## [1.2.0] - 2023-06-08
### Added
* Various Auto3DSeg enhancements and integration tests including multi-node multi-GPU optimization, major usability improvements
* TensorRT and ONNX support for `monai.bundle` API and the relevant models
* nnU-Net V2 integration `monai.apps.nnunet`
* Binary and categorical metrics and event handlers using `MetricsReloaded`
* Python module and CLI entry point for bundle workflows in `monai.bundle.workflows` and `monai.fl.client`
* Modular patch inference API including `PatchInferer`, `merger`, and `splitter`
* Initial release of lazy resampling including transforms and MetaTensor implementations
* Bridge for ITK Image object and MetaTensor `monai.data.itk_torch_bridge`
* Sliding window inference memory efficiency optimization including `SlidingWindowInfererAdapt`
* Generic kernel filtering transforms `ImageFiltered` and `RandImageFiltered`
* Trainable bilateral filters and joint bilateral filters
* ClearML stats and image handlers for experiment tracking
#### misc.
* Utility functions to warn API default value changes (#5738)
* Support of dot notation to access content of `ConfigParser` (#5813)
* Softmax version to focal loss (#6544)
* FROC metric for N-dimensional (#6528)
* Extend SurfaceDiceMetric for 3D images (#6549)
* A `track_meta` option for Lambda and derived transforms (#6385)
* CLIP pre-trained text-to-vision embedding (#6282)
* Optional spacing to surface distances calculations (#6144)
* `WSIReader` read by power and mpp (#6244)
* Support GPU tensor for `GridPatch` and `GridPatchDataset` (#6246)
* `SomeOf` transform composer (#6143)
* GridPatch with both count and threshold filtering (#6055)
### Fixed
#### transforms
* `map_classes_to_indices` efficiency issue (#6468)
* Adaptive resampling mode based on backends (#6429)
* Improve Compose encapsulation (#6224)
* User-provided `FolderLayout` in `SaveImage` and `SaveImaged` transforms (#6213)
* `SpacingD` output shape compute stability (#6126)
* No mutate ratio /user inputs `croppad` (#6127)
* A `warn` flag to RandCropByLabelClasses (#6121)
* `nan` to indicate `no_channel`, split dim singleton (#6090)
* Compatible padding mode (#6076)
* Allow for missing `filename_or_obj` key (#5980)
* `Spacing` pixdim in-place change (#5950)
* Add warning in `RandHistogramShift` (#5877)
* Exclude `cuCIM` wrappers from `get_transform_backends` (#5838)
#### data
* `__format__` implementation of MetaTensor (#6523)
* `channel_dim` in `TiffFileWSIReader` and `CuCIMWSIReader` (#6514)
* Prepend `"meta"` to `MetaTensor.__repr__` and `MetaTensor.__str__` for easier identification (#6214)
* MetaTensor slicing issue (#5845)
* Default writer flags (#6147)
* `WSIReader` defaults and tensor conversion (#6058)
* Remove redundant array copy for WSITiffFileReader (#6089)
* Fix unused arg in `SlidingPatchWSIDataset` (#6047)
* `reverse_indexing` for PILReader (#6008)
* Use `np.linalg` for the small affine inverse (#5967)
#### metrics and losses
* Removing L2-norm in contrastive loss (L2-norm already present in CosSim) (#6550)
* Fixes the SSIM metric (#6250)
* Efficiency issues of Dice metrics (#6412)
* Generalized Dice issue (#5929)
* Unify output tensor devices for multiple metrics (#5924)
#### networks
* Make `RetinaNet` throw errors for NaN only when training (#6479)
* Replace deprecated arg in torchvision models (#6401)
* Improves NVFuser import check (#6399)
* Add `device` in `HoVerNetNuclearTypePostProcessing` and `HoVerNetInstanceMapPostProcessing` (#6333)
* Enhance hovernet load pretrained function (#6269)
* Access to the `att_mat` in self-attention modules (#6493)
* Optional swinunetr-v2 (#6203)
* Add transform to handle empty box as training data for `retinanet_detector` (#6170)
* GPU utilization of DiNTS network (#6050)
* A pixelshuffle upsample shape mismatch problem (#5982)
* GEGLU activation function for the MLP Block (#5856)
* Constructors for `DenseNet` derived classes (#5846)
* Flexible interpolation modes in `regunet` (#5807)
#### bundle
* Optimized the `deepcopy` logic in `ConfigParser` (#6464)
* Improve check and error message of bundle run (#6400)
* Warn or raise ValueError on duplicated key in json/yaml config (#6252)
* Default metadata and logging values for bundle run (#6072)
* `pprint` head and tail in bundle script (#5969)
* Config parsing issue for substring reference (#5932)
* Fix instantiate for object instantiation with attribute `path` (#5866)
* Fix `_get_latest_bundle_version` issue on Windows (#5787)
#### engines and handlers
* MLflow handler run bug (#6446)
* `monai.engine` training attribute check (#6132)
* Update StatsHandler logging message (#6051)
* Added callable options for `iteration_log` and `epoch_log` in TensorBoard and MLFlow (#5976)
* `CheckpointSaver` logging error (#6026)
* Callable options for `iteration_log` and `epoch_log` in StatsHandler (#5965)
#### misc.
* Avoid creating cufile.log when `import monai` (#6106)
* `monai._extensions` module compatibility with rocm (#6161)
* Issue of repeated UserWarning: "TypedStorage is deprecated" (#6105)
* Use logging config at module level (#5960)
* Add ITK to the list of optional dependencies (#5858)
* `RankFilter` to skip logging when the rank is not meeting criteria (#6243)
* Various documentation issues
### Changed
* Overall more precise and consistent type annotations
* Optionally depend on PyTorch-Ignite v0.4.11 instead of v0.4.10
* Base Docker image upgraded to `nvcr.io/nvidia/pytorch:23.03-py3` from `nvcr.io/nvidia/pytorch:22.10-py3`
### Deprecated
* `resample=True`; `resample=False` will be the new default in `SaveImage`
* `random_size=True`; `random_size=False` will be the new default for the random cropping transforms
* `image_only=False`; `image_only=True` will be the new default in `LoadImage`
* `AddChannel` and `AsChannelFirst` in favor of `EnsureChannelFirst`
### Removed
* Deprecated APIs since v0.9, including WSIReader from `monai.apps`, `NiftiSaver` and `PNGSaver` from `monai.data`
* Support for PyTorch 1.8
* Support for Python 3.7

## [1.1.0] - 2022-12-19
### Added
* Hover-Net based digital pathology workflows including new network, loss, postprocessing, metric, training, and inference modules
* Various enhancements for Auto3dSeg `AutoRunner` including template caching, selection, and a dry-run mode `nni_dry_run`
* Various enhancements for Auto3dSeg algo templates including new state-of-the-art configurations, optimized GPU memory utilization
* New bundle API and configurations to support experiment management including `MLFlowHandler`
* New `bundle.script` API to support model zoo query and download
* `LossMetric` metric to compute loss as cumulative metric measurement
* Transforms and base transform APIs including `RandomizableTrait` and `MedianSmooth`
* `runtime_cache` option for `CacheDataset` and the derived classes to allow for shared caching on the fly
* Flexible name formatter for `SaveImage` transform
* `pending_operations` MetaTensor property and basic APIs for lazy image resampling
* Contrastive sensitivity for SSIM metric
* Extensible backbones for `FlexibleUNet`
* Generalize `SobelGradients` to 3D and any spatial axes
* `warmup_multiplier` option for `WarmupCosineSchedule`
* F beta score metric based on confusion matrix metric
* Support of key overwriting in `Lambdad`
* Basic premerge tests for Python 3.11
* Unit and integration tests for CUDA 11.6, 11.7 and A100 GPU
* `DataAnalyzer` handles minor image-label shape inconsistencies
### Fixed
* Review and enhance previously untyped APIs with additional type annotations and casts
* `switch_endianness` in LoadImage now supports tensor input
* Reduced memory footprint for various Auto3dSeg tests
* Issue of `@` in `monai.bundle.ReferenceResolver`
* Compatibility issue with ITK-Python 5.3 (converting `itkMatrixF44` for default collate)
* Inconsistent of sform and qform when using different backends for `SaveImage`
* `MetaTensor.shape` call now returns a `torch.Size` instead of tuple
* Issue of channel reduction in `GeneralizedDiceLoss`
* Issue of background handling before softmax in `DiceFocalLoss`
* Numerical issue of `LocalNormalizedCrossCorrelationLoss`
* Issue of incompatible view size in `ConfusionMatrixMetric`
* `NetAdapter` compatibility with Torchscript
* Issue of `extract_levels` in `RegUNet`
* Optional `bias_downsample` in `ResNet`
* `dtype` overflow for `ShiftIntensity` transform
* Randomized transforms such as `RandCuCIM` now inherit `RandomizableTrait`
* `fg_indices.size` compatibility issue in `generate_pos_neg_label_crop_centers`
* Issue when inverting `ToTensor`
* Issue of capital letters in filename suffixes check in `LoadImage`
* Minor tensor compatibility issues in `apps.nuclick.transforms`
* Issue of float16 in `verify_net_in_out`
* `std` variable type issue for `RandRicianNoise`
* `DataAnalyzer` accepts `None` as label key and checks empty labels
* `iter_patch_position` now has a smaller memory footprint
* `CumulativeAverage` has been refactored and enhanced to allow for simple tracking of metric running stats.
* Multi-threading issue for `MLFlowHandler`
### Changed
* Printing a MetaTensor now generates a less verbose representation
* `DistributedSampler` raises a ValueError if there are too few devices
* OpenCV and `VideoDataset` modules are loaded lazily to avoid dependency issues
* `device` in `monai.engines.Workflow` supports string values
* `Activations` and `AsDiscrete` take `kwargs` as additional arguments
* `DataAnalyzer` is now more efficient and writes summary stats before detailed all case stats
* Base Docker image upgraded to `nvcr.io/nvidia/pytorch:22.10-py3` from `nvcr.io/nvidia/pytorch:22.09-py3`
* Simplified Conda environment file `environment-dev.yml`
* Versioneer dependency upgraded to `0.23` from `0.19`
### Deprecated
* `NibabelReader` input argument `dtype` is deprecated, the reader will use the original dtype of the image
### Removed
* Support for PyTorch 1.7

## [1.0.1] - 2022-10-24
### Fixes
* DiceCELoss for multichannel targets
* Auto3DSeg DataAnalyzer out-of-memory error and other minor issues
* An optional flag issue in the RetinaNet detector
* An issue with output offset for Spacing
* A `LoadImage` issue when `track_meta` is `False`
* 1D data output error in `VarAutoEncoder`
* An issue with resolution computing in `ImageStats`
### Added
* Flexible min/max pixdim options for Spacing
* Upsample mode `deconvgroup` and optional kernel sizes
* Docstrings for gradient-based saliency maps
* Occlusion sensitivity to use sliding window inference
* Enhanced Gaussian window and device assignments for sliding window inference
* Multi-GPU support for MonaiAlgo
* `ClientAlgoStats` and `MonaiAlgoStats` for federated summary statistics
* MetaTensor support for `OneOf`
* Add a file check for bundle logging config
* Additional content and an authentication token option for bundle info API
* An anti-aliasing option for `Resized`
* `SlidingWindowInferer` adaptive device based on `cpu_thresh`
* `SegResNetDS` with deep supervision and non-isotropic kernel support
* Premerge tests for Python 3.10
### Changed
* Base Docker image upgraded to `nvcr.io/nvidia/pytorch:22.09-py3` from `nvcr.io/nvidia/pytorch:22.08-py3`
* Replace `None` type metadata content with `"none"` for `collate_fn` compatibility
* HoVerNet Mode and Branch to independent StrEnum
* Automatically infer device from the first item in random elastic deformation dict
* Add channel dim in `ComputeHoVerMaps` and `ComputeHoVerMapsd`
* Remove batch dim in `SobelGradients` and `SobelGradientsd`
### Deprecated
* Deprecating `compute_meandice`, `compute_meaniou` in `monai.metrics`, in favor of
`compute_dice` and `compute_iou` respectively

## [1.0.0] - 2022-09-16
### Added
* `monai.auto3dseg` base APIs and `monai.apps.auto3dseg` components for automated machine learning (AutoML) workflow
* `monai.fl` module with base APIs and `MonaiAlgo` for federated learning client workflow
* An initial backwards compatibility [guide](https://github.com/Project-MONAI/MONAI/blob/dev/CONTRIBUTING.md#backwards-compatibility)
* Initial release of accelerated MRI reconstruction components, including `CoilSensitivityModel`
* Support of `MetaTensor` and new metadata attributes for various digital pathology components
* Various `monai.bundle` enhancements for MONAI model-zoo usability, including config debug mode and `get_all_bundles_list`
* new `monai.transforms` components including `SignalContinuousWavelet` for 1D signal, `ComputeHoVerMaps` for digital pathology, and `SobelGradients` for spatial gradients
* `VarianceMetric` and `LabelQualityScore` metrics for active learning
* Dataset API for real-time stream and videos
* Several networks and building blocks including `FlexibleUNet` and `HoVerNet`
* `MeanIoUHandler` and `LogfileHandler` workflow event handlers
* `WSIReader` with the TiffFile backend
* Multi-threading in `WSIReader` with cuCIM backend
* `get_stats` API in `monai.engines.Workflow`
* `prune_meta_pattern` in `monai.transforms.LoadImage`
* `max_interactions` for deepedit interaction workflow
* Various profiling utilities in `monai.utils.profiling`
### Changed
* Base Docker image upgraded to `nvcr.io/nvidia/pytorch:22.08-py3` from `nvcr.io/nvidia/pytorch:22.06-py3`
* Optionally depend on PyTorch-Ignite v0.4.10 instead of v0.4.9
* The cache-based dataset now matches the transform information when read/write the cache
* `monai.losses.ContrastiveLoss` now infers `batch_size` during `forward()`
* Rearrange the spatial axes in `RandSmoothDeform` transforms following PyTorch's convention
* Unified several environment flags into `monai.utils.misc.MONAIEnvVars`
* Simplified `__str__` implementation of `MetaTensor` instead of relying on the `__repr__` implementation
### Fixed
* Improved error messages when both `monai` and `monai-weekly` are pip-installed
* Inconsistent pseudo number sequences for different `num_workers` in `DataLoader`
* Issue of repeated sequences for `monai.data.ShuffleBuffer`
* Issue of not preserving the physical extent in `monai.transforms.Spacing`
* Issue of using `inception_v3` as the backbone of `monai.networks.nets.TorchVisionFCModel`
* Index device issue for `monai.transforms.Crop`
* Efficiency issue when converting the array dtype and contiguous memory
### Deprecated
* `Addchannel` and `AsChannelFirst` transforms in favor of `EnsureChannelFirst`
* `monai.apps.pathology.data` components in favor of the corresponding components from `monai.data`
* `monai.apps.pathology.handlers` in favor of the corresponding components from `monai.handlers`
### Removed
* `Status` section in the pull request template in favor of the pull request draft mode
* `monai.engines.BaseWorkflow`
* `ndim` and `dimensions` arguments in favor of `spatial_dims`
* `n_classes`, `num_classes` arguments in `AsDiscrete` in favor of `to_onehot`
* `logit_thresh`, `threshold_values` arguments in `AsDiscrete` in favor of `threshold`
* `torch.testing.assert_allclose` in favor of `tests.utils.assert_allclose`

## [0.9.1] - 2022-07-22
### Added
* Support of `monai.data.MetaTensor` as core data structure across the modules
* Support of `inverse` in array-based transforms
* `monai.apps.TciaDataset` APIs for The Cancer Imaging Archive (TCIA) datasets, including a pydicom-backend reader
* Initial release of components for MRI reconstruction in `monai.apps.reconstruction`, including various FFT utilities
* New metrics and losses, including mean IoU and structural similarity index
* `monai.utils.StrEnum` class to simplify Enum-based type annotations
### Changed
* Base Docker image upgraded to `nvcr.io/nvidia/pytorch:22.06-py3` from `nvcr.io/nvidia/pytorch:22.04-py3`
* Optionally depend on PyTorch-Ignite v0.4.9 instead of v0.4.8
### Fixed
* Fixed issue of not skipping post activations in `Convolution` when input arguments are None
* Fixed issue of ignoring dropout arguments in `DynUNet`
* Fixed issue of hard-coded non-linear function in ViT classification head
* Fixed issue of in-memory config overriding with `monai.bundle.ConfigParser.update`
* 2D SwinUNETR incompatible shapes
* Fixed issue with `monai.bundle.verify_metadata` not raising exceptions
* Fixed issue with `monai.transforms.GridPatch` returns inconsistent type location when padding
* Wrong generalized Dice score metric when denominator is 0 but prediction is non-empty
* Docker image build error due to NGC CLI upgrade
* Optional default value when parsing id unavailable in a ConfigParser instance
* Immutable data input for the patch-based WSI datasets
### Deprecated
* `*_transforms` and `*_meta_dict` fields in dictionary-based transforms in favor of MetaTensor
* `meta_keys`, `meta_key_postfix`, `src_affine` arguments in various transforms, in favor of MetaTensor
* `AsChannelFirst` and `AddChannel`, in favor of `EnsureChannelFirst` transform

## [0.9.0] - 2022-06-08
### Added
* `monai.bundle` primary module with a `ConfigParser` and command-line interfaces for configuration-based workflows
* Initial release of MONAI bundle specification
* Initial release of volumetric image detection modules including bounding boxes handling, RetinaNet-based architectures
* API preview `monai.data.MetaTensor`
* Unified `monai.data.image_writer` to support flexible IO backends including an ITK writer
* Various new network blocks and architectures including `SwinUNETR`
* DeepEdit interactive training/validation workflow
* NuClick interactive segmentation transforms
* Patch-based readers and datasets for whole-slide imaging
* New losses and metrics including `SurfaceDiceMetric`, `GeneralizedDiceFocalLoss`
* New pre-processing transforms including `RandIntensityRemap`, `SpatialResample`
* Multi-output and slice-based inference for `SlidingWindowInferer`
* `NrrdReader` for NRRD file support
* Torchscript utilities to save models with meta information
* Gradient-based visualization module `SmoothGrad`
* Automatic regular source code scanning for common vulnerabilities and coding errors

### Changed
* Simplified `TestTimeAugmentation` using de-collate and invertible transforms APIs
* Refactoring `monai.apps.pathology` modules into `monai.handlers` and `monai.transforms`
* Flexible activation and normalization layers for `TopologySearch` and `DiNTS`
* Anisotropic first layers for 3D resnet
* Flexible ordering of activation, normalization in `UNet`
* Enhanced performance of connected-components analysis using Cupy
* `INSTANCE_NVFUSER` for enhanced performance in 3D instance norm
* Support of string representation of dtype in `convert_data_type`
* Added new options `iteration_log`, `iteration_log` to the logging handlers
* Base Docker image upgraded to `nvcr.io/nvidia/pytorch:22.04-py3` from `nvcr.io/nvidia/pytorch:21.10-py3`
* `collate_fn` generates more data-related debugging info with `dev_collate`

### Fixed
* Unified the spellings of "meta data", "metadata", "meta-data" to "metadata"
* Various inaccurate error messages when input data are in invalid shapes
* Issue of computing symmetric distances in `compute_average_surface_distance`
* Unnecessary layer  `self.conv3` in `UnetResBlock`
* Issue of torchscript compatibility for `ViT` and self-attention blocks
* Issue of hidden layers in `UNETR`
* `allow_smaller` in spatial cropping transforms
* Antialiasing in `Resize`
* Issue of bending energy loss value at different resolutions
* `kwargs_read_csv` in `CSVDataset`
* In-place modification in `Metric` reduction
* `wrap_array` for `ensure_tuple`
* Contribution guide for introducing new third-party dependencies

### Removed
* Deprecated `nifti_writer`, `png_writer` in favor of `monai.data.image_writer`
* Support for PyTorch 1.6

## [0.8.1] - 2022-02-16
### Added
* Support of `matshow3d` with given `channel_dim`
* Support of spatial 2D for `ViTAutoEnc`
* Support of `dataframe` object input in `CSVDataset`
* Support of tensor backend for `Orientation`
* Support of configurable delimiter for CSV writers
* A base workflow API
* `DataFunc` API for dataset-level preprocessing
* `write_scalar` API for logging with additional `engine` parameter in `TensorBoardHandler`
* Enhancements for NVTX Range transform logging
* Enhancements for `set_determinism`
* Performance enhancements in the cache-based datasets
* Configurable metadata keys for `monai.data.DatasetSummary`
* Flexible `kwargs` for `WSIReader`
* Logging for the learning rate schedule handler
* `GridPatchDataset` as subclass of `monai.data.IterableDataset`
* `is_onehot` option in `KeepLargestConnectedComponent`
* `channel_dim` in the image readers and support of stacking images with channels
* Skipping workflow `run` if epoch length is 0
* Enhanced `CacheDataset` to avoid duplicated cache items
* `save_state` utility function

### Changed
* Optionally depend on PyTorch-Ignite v0.4.8 instead of v0.4.6
* `monai.apps.mmars.load_from_mmar` defaults to the latest version

### Fixed
* Issue when caching large items with `pickle`
* Issue of hard-coded activation functions in `ResBlock`
* Issue of `create_file_name` assuming local disk file creation
* Issue of `WSIReader` when the backend is `TiffFile`
* Issue of `deprecated_args` when the function signature contains kwargs
* Issue of `channel_wise` computations for the intensity-based transforms
* Issue of inverting `OneOf`
* Issue of removing temporary caching file for the persistent dataset
* Error messages when reader backend is not available
* Output type casting issue in `ScaleIntensityRangePercentiles`
* Various docstring typos and broken URLs
* `mode` in the evaluator engine
* Ordering of `Orientation` and `Spacing` in `monai.apps.deepgrow.dataset`

### Removed
* Additional deep supervision modules in `DynUnet`
* Deprecated `reduction` argument for `ContrastiveLoss`
* Decollate warning in `Workflow`
* Unique label exception in `ROCAUCMetric`
* Logger configuration logic in the event handlers

## [0.8.0] - 2021-11-25
### Added
* Overview of [new features in v0.8](docs/source/whatsnew_0_8.md)
* Network modules for differentiable neural network topology search (DiNTS)
* Multiple Instance Learning transforms and models for digital pathology WSI analysis
* Vision transformers for self-supervised representation learning
* Contrastive loss for self-supervised learning
* Finalized major improvements of 200+ components in `monai.transforms` to support input and backend in PyTorch and NumPy
* Initial registration module benchmarking with `GlobalMutualInformationLoss` as an example
* `monai.transforms` documentation with visual examples and the utility functions
* Event handler for `MLfLow` integration
* Enhanced data visualization functions including `blend_images` and `matshow3d`
* `RandGridDistortion` and `SmoothField` in `monai.transforms`
* Support of randomized shuffle buffer in iterable datasets
* Performance review and enhancements for data type casting
* Cumulative averaging API with distributed environment support
* Module utility functions including `require_pkg` and `pytorch_after`
* Various usability enhancements such as `allow_smaller` when sampling ROI and `wrap_sequence` when casting object types
* `tifffile` support in `WSIReader`
* Regression tests for the fast training workflows
* Various tutorials and demos including educational contents at [MONAI Bootcamp 2021](https://github.com/Project-MONAI/MONAIBootcamp2021)
### Changed
* Base Docker image upgraded to `nvcr.io/nvidia/pytorch:21.10-py3` from `nvcr.io/nvidia/pytorch:21.08-py3`
* Decoupled `TraceKeys` and `TraceableTransform` APIs from `InvertibleTransform`
* Skipping affine-based resampling when `resample=False` in `NiftiSaver`
* Deprecated `threshold_values: bool` and `num_classes: int` in `AsDiscrete`
* Enhanced `apply_filter` for spatially 1D, 2D and 3D inputs with non-separable kernels
* Logging with `logging` in downloading and model archives in `monai.apps`
* API documentation site now defaults to `stable` instead of `latest`
* `skip-magic-trailing-comma` in coding style enforcements
* Pre-merge CI pipelines now include unit tests with Nvidia Ampere architecture
### Removed
* Support for PyTorch 1.5
* The deprecated `DynUnetV1` and the related network blocks
* GitHub self-hosted CI/CD pipelines for package releases
### Fixed
* Support of path-like objects as file path inputs in most modules
* Issue of `decollate_batch` for dictionary of empty lists
* Typos in documentation and code examples in various modules
* Issue of no available keys when `allow_missing_keys=True` for the `MapTransform`
* Issue of redundant computation when normalization factors are 0.0 and 1.0 in `ScaleIntensity`
* Incorrect reports of registered readers in `ImageReader`
* Wrong numbering of iterations in `StatsHandler`
* Naming conflicts in network modules and aliases
* Incorrect output shape when `reduction="none"` in `FocalLoss`
* Various usability issues reported by users

## [0.7.0] - 2021-09-24
### Added
* Overview of [new features in v0.7](docs/source/whatsnew_0_7.md)
* Initial phase of major usability improvements in `monai.transforms` to support input and backend in PyTorch and NumPy
* Performance enhancements, with [profiling and tuning guides](https://github.com/Project-MONAI/tutorials/blob/master/acceleration/fast_model_training_guide.md) for typical use cases
* Reproducing [training modules and workflows](https://github.com/Project-MONAI/tutorials/tree/master/kaggle/RANZCR/4th_place_solution) of state-of-the-art Kaggle competition solutions
* 24 new transforms, including
  * `OneOf` meta transform
  * DeepEdit guidance signal transforms for interactive segmentation
  * Transforms for self-supervised pre-training
  * Integration of [NVIDIA Tools Extension](https://developer.nvidia.com/blog/nvidia-tools-extension-api-nvtx-annotation-tool-for-profiling-code-in-python-and-c-c/) (NVTX)
  * Integration of [cuCIM](https://github.com/rapidsai/cucim)
  * Stain normalization and contextual grid for digital pathology
* `Transchex` network for vision-language transformers for chest X-ray analysis
* `DatasetSummary` utility in `monai.data`
* `WarmupCosineSchedule`
* Deprecation warnings and documentation support for better backwards compatibility
* Padding with additional `kwargs` and different backend API
* Additional options such as `dropout` and `norm` in various networks and their submodules

### Changed
* Base Docker image upgraded to `nvcr.io/nvidia/pytorch:21.08-py3` from `nvcr.io/nvidia/pytorch:21.06-py3`
* Deprecated input argument `n_classes`, in favor of `num_classes`
* Deprecated input argument `dimensions` and `ndims`, in favor of `spatial_dims`
* Updated the Sphinx-based documentation theme for better readability
* `NdarrayTensor` type is replaced by `NdarrayOrTensor` for simpler annotations
* Self-attention-based network blocks now support both 2D and 3D inputs

### Removed
* The deprecated `TransformInverter`, in favor of `monai.transforms.InvertD`
* GitHub self-hosted CI/CD pipelines for nightly and post-merge tests
* `monai.handlers.utils.evenly_divisible_all_gather`
* `monai.handlers.utils.string_list_all_gather`

### Fixed
* A Multi-thread cache writing issue in `LMDBDataset`
* Output shape convention inconsistencies of the image readers
* Output directory and file name flexibility issue for `NiftiSaver`, `PNGSaver`
* Requirement of the `label` field in test-time augmentation
* Input argument flexibility issues for  `ThreadDataLoader`
* Decoupled `Dice` and `CrossEntropy` intermediate results in `DiceCELoss`
* Improved documentation, code examples, and warning messages in various modules
* Various usability issues reported by users

## [0.6.0] - 2021-07-08
### Added
* 10 new transforms, a masked loss wrapper, and a `NetAdapter` for transfer learning
* APIs to load networks and pre-trained weights from Clara Train [Medical Model ARchives (MMARs)](https://docs.nvidia.com/clara/clara-train-sdk/pt/mmar.html)
* Base metric and cumulative metric APIs, 4 new regression metrics
* Initial CSV dataset support
* Decollating mini-batch as the default first postprocessing step, [Migrating your v0.5 code to v0.6](https://github.com/Project-MONAI/MONAI/wiki/v0.5-to-v0.6-migration-guide) wiki shows how to adapt to the breaking changes
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

[Unreleased]: https://github.com/Project-MONAI/MONAI/compare/1.3.2...HEAD
[1.3.2]: https://github.com/Project-MONAI/MONAI/compare/1.3.1...1.3.2
[1.3.1]: https://github.com/Project-MONAI/MONAI/compare/1.3.0...1.3.1
[1.3.0]: https://github.com/Project-MONAI/MONAI/compare/1.2.0...1.3.0
[1.2.0]: https://github.com/Project-MONAI/MONAI/compare/1.1.0...1.2.0
[1.1.0]: https://github.com/Project-MONAI/MONAI/compare/1.0.1...1.1.0
[1.0.1]: https://github.com/Project-MONAI/MONAI/compare/1.0.0...1.0.1
[1.0.0]: https://github.com/Project-MONAI/MONAI/compare/0.9.1...1.0.0
[0.9.1]: https://github.com/Project-MONAI/MONAI/compare/0.9.0...0.9.1
[0.9.0]: https://github.com/Project-MONAI/MONAI/compare/0.8.1...0.9.0
[0.8.1]: https://github.com/Project-MONAI/MONAI/compare/0.8.0...0.8.1
[0.8.0]: https://github.com/Project-MONAI/MONAI/compare/0.7.0...0.8.0
[0.7.0]: https://github.com/Project-MONAI/MONAI/compare/0.6.0...0.7.0
[0.6.0]: https://github.com/Project-MONAI/MONAI/compare/0.5.3...0.6.0
[0.5.3]: https://github.com/Project-MONAI/MONAI/compare/0.5.0...0.5.3
[0.5.0]: https://github.com/Project-MONAI/MONAI/compare/0.4.0...0.5.0
[0.4.0]: https://github.com/Project-MONAI/MONAI/compare/0.3.0...0.4.0
[0.3.0]: https://github.com/Project-MONAI/MONAI/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/Project-MONAI/MONAI/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/Project-MONAI/MONAI/commits/0.1.0
