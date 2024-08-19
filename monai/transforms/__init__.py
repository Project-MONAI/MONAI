# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from .adaptors import FunctionSignature, adaptor, apply_alias, to_kwargs
from .compose import Compose, OneOf, RandomOrder, SomeOf
from .croppad.array import (
    BorderPad,
    BoundingRect,
    CenterScaleCrop,
    CenterSpatialCrop,
    Crop,
    CropForeground,
    DivisiblePad,
    Pad,
    RandCropByLabelClasses,
    RandCropByPosNegLabel,
    RandScaleCrop,
    RandSpatialCrop,
    RandSpatialCropSamples,
    RandWeightedCrop,
    ResizeWithPadOrCrop,
    SpatialCrop,
    SpatialPad,
)
from .croppad.batch import PadListDataCollate
from .croppad.dictionary import (
    BorderPadd,
    BorderPadD,
    BorderPadDict,
    BoundingRectd,
    BoundingRectD,
    BoundingRectDict,
    CenterScaleCropd,
    CenterScaleCropD,
    CenterScaleCropDict,
    CenterSpatialCropd,
    CenterSpatialCropD,
    CenterSpatialCropDict,
    Cropd,
    CropD,
    CropDict,
    CropForegroundd,
    CropForegroundD,
    CropForegroundDict,
    DivisiblePadd,
    DivisiblePadD,
    DivisiblePadDict,
    Padd,
    PadD,
    PadDict,
    RandCropByLabelClassesd,
    RandCropByLabelClassesD,
    RandCropByLabelClassesDict,
    RandCropByPosNegLabeld,
    RandCropByPosNegLabelD,
    RandCropByPosNegLabelDict,
    RandCropd,
    RandCropD,
    RandCropDict,
    RandScaleCropd,
    RandScaleCropD,
    RandScaleCropDict,
    RandSpatialCropd,
    RandSpatialCropD,
    RandSpatialCropDict,
    RandSpatialCropSamplesd,
    RandSpatialCropSamplesD,
    RandSpatialCropSamplesDict,
    RandWeightedCropd,
    RandWeightedCropD,
    RandWeightedCropDict,
    ResizeWithPadOrCropd,
    ResizeWithPadOrCropD,
    ResizeWithPadOrCropDict,
    SpatialCropd,
    SpatialCropD,
    SpatialCropDict,
    SpatialPadd,
    SpatialPadD,
    SpatialPadDict,
)
from .croppad.functional import crop_func, crop_or_pad_nd, pad_func, pad_nd
from .intensity.array import (
    AdjustContrast,
    ClipIntensityPercentiles,
    ComputeHoVerMaps,
    DetectEnvelope,
    ForegroundMask,
    GaussianSharpen,
    GaussianSmooth,
    GibbsNoise,
    HistogramNormalize,
    IntensityRemap,
    KSpaceSpikeNoise,
    MaskIntensity,
    MedianSmooth,
    NormalizeIntensity,
    RandAdjustContrast,
    RandBiasField,
    RandCoarseDropout,
    RandCoarseShuffle,
    RandCoarseTransform,
    RandGaussianNoise,
    RandGaussianSharpen,
    RandGaussianSmooth,
    RandGibbsNoise,
    RandHistogramShift,
    RandIntensityRemap,
    RandKSpaceSpikeNoise,
    RandRicianNoise,
    RandScaleIntensity,
    RandScaleIntensityFixedMean,
    RandShiftIntensity,
    RandStdShiftIntensity,
    SavitzkyGolaySmooth,
    ScaleIntensity,
    ScaleIntensityFixedMean,
    ScaleIntensityRange,
    ScaleIntensityRangePercentiles,
    ShiftIntensity,
    StdShiftIntensity,
    ThresholdIntensity,
    UltrasoundConfidenceMapTransform,
)
from .intensity.dictionary import (
    AdjustContrastd,
    AdjustContrastD,
    AdjustContrastDict,
    ClipIntensityPercentilesd,
    ClipIntensityPercentilesD,
    ClipIntensityPercentilesDict,
    ComputeHoVerMapsd,
    ComputeHoVerMapsD,
    ComputeHoVerMapsDict,
    ForegroundMaskd,
    ForegroundMaskD,
    ForegroundMaskDict,
    GaussianSharpend,
    GaussianSharpenD,
    GaussianSharpenDict,
    GaussianSmoothd,
    GaussianSmoothD,
    GaussianSmoothDict,
    GibbsNoised,
    GibbsNoiseD,
    GibbsNoiseDict,
    HistogramNormalized,
    HistogramNormalizeD,
    HistogramNormalizeDict,
    KSpaceSpikeNoised,
    KSpaceSpikeNoiseD,
    KSpaceSpikeNoiseDict,
    MaskIntensityd,
    MaskIntensityD,
    MaskIntensityDict,
    MedianSmoothd,
    MedianSmoothD,
    MedianSmoothDict,
    NormalizeIntensityd,
    NormalizeIntensityD,
    NormalizeIntensityDict,
    RandAdjustContrastd,
    RandAdjustContrastD,
    RandAdjustContrastDict,
    RandBiasFieldd,
    RandBiasFieldD,
    RandBiasFieldDict,
    RandCoarseDropoutd,
    RandCoarseDropoutD,
    RandCoarseDropoutDict,
    RandCoarseShuffled,
    RandCoarseShuffleD,
    RandCoarseShuffleDict,
    RandGaussianNoised,
    RandGaussianNoiseD,
    RandGaussianNoiseDict,
    RandGaussianSharpend,
    RandGaussianSharpenD,
    RandGaussianSharpenDict,
    RandGaussianSmoothd,
    RandGaussianSmoothD,
    RandGaussianSmoothDict,
    RandGibbsNoised,
    RandGibbsNoiseD,
    RandGibbsNoiseDict,
    RandHistogramShiftd,
    RandHistogramShiftD,
    RandHistogramShiftDict,
    RandKSpaceSpikeNoised,
    RandKSpaceSpikeNoiseD,
    RandKSpaceSpikeNoiseDict,
    RandRicianNoised,
    RandRicianNoiseD,
    RandRicianNoiseDict,
    RandScaleIntensityd,
    RandScaleIntensityD,
    RandScaleIntensityDict,
    RandScaleIntensityFixedMeand,
    RandScaleIntensityFixedMeanD,
    RandScaleIntensityFixedMeanDict,
    RandShiftIntensityd,
    RandShiftIntensityD,
    RandShiftIntensityDict,
    RandStdShiftIntensityd,
    RandStdShiftIntensityD,
    RandStdShiftIntensityDict,
    SavitzkyGolaySmoothd,
    SavitzkyGolaySmoothD,
    SavitzkyGolaySmoothDict,
    ScaleIntensityd,
    ScaleIntensityD,
    ScaleIntensityDict,
    ScaleIntensityRanged,
    ScaleIntensityRangeD,
    ScaleIntensityRangeDict,
    ScaleIntensityRangePercentilesd,
    ScaleIntensityRangePercentilesD,
    ScaleIntensityRangePercentilesDict,
    ShiftIntensityd,
    ShiftIntensityD,
    ShiftIntensityDict,
    StdShiftIntensityd,
    StdShiftIntensityD,
    StdShiftIntensityDict,
    ThresholdIntensityd,
    ThresholdIntensityD,
    ThresholdIntensityDict,
)
from .inverse import InvertibleTransform, TraceableTransform
from .inverse_batch_transform import BatchInverseTransform, Decollated, DecollateD, DecollateDict
from .io.array import SUPPORTED_READERS, LoadImage, SaveImage, WriteFileMapping
from .io.dictionary import LoadImaged, LoadImageD, LoadImageDict, SaveImaged, SaveImageD, SaveImageDict
from .lazy.array import ApplyPending
from .lazy.dictionary import ApplyPendingd, ApplyPendingD, ApplyPendingDict
from .lazy.functional import apply_pending
from .lazy.utils import combine_transforms, resample
from .meta_utility.dictionary import (
    FromMetaTensord,
    FromMetaTensorD,
    FromMetaTensorDict,
    ToMetaTensord,
    ToMetaTensorD,
    ToMetaTensorDict,
)
from .nvtx import (
    Mark,
    Markd,
    MarkD,
    MarkDict,
    RandMark,
    RandMarkd,
    RandMarkD,
    RandMarkDict,
    RandRangePop,
    RandRangePopd,
    RandRangePopD,
    RandRangePopDict,
    RandRangePush,
    RandRangePushd,
    RandRangePushD,
    RandRangePushDict,
    RangePop,
    RangePopd,
    RangePopD,
    RangePopDict,
    RangePush,
    RangePushd,
    RangePushD,
    RangePushDict,
)
from .post.array import (
    Activations,
    AsDiscrete,
    DistanceTransformEDT,
    FillHoles,
    Invert,
    KeepLargestConnectedComponent,
    LabelFilter,
    LabelToContour,
    MeanEnsemble,
    ProbNMS,
    RemoveSmallObjects,
    SobelGradients,
    VoteEnsemble,
)
from .post.dictionary import (
    ActivationsD,
    Activationsd,
    ActivationsDict,
    AsDiscreteD,
    AsDiscreted,
    AsDiscreteDict,
    DistanceTransformEDTd,
    DistanceTransformEDTD,
    DistanceTransformEDTDict,
    Ensembled,
    EnsembleD,
    EnsembleDict,
    FillHolesD,
    FillHolesd,
    FillHolesDict,
    InvertD,
    Invertd,
    InvertDict,
    KeepLargestConnectedComponentD,
    KeepLargestConnectedComponentd,
    KeepLargestConnectedComponentDict,
    LabelFilterD,
    LabelFilterd,
    LabelFilterDict,
    LabelToContourD,
    LabelToContourd,
    LabelToContourDict,
    MeanEnsembleD,
    MeanEnsembled,
    MeanEnsembleDict,
    ProbNMSD,
    ProbNMSd,
    ProbNMSDict,
    RemoveSmallObjectsD,
    RemoveSmallObjectsd,
    RemoveSmallObjectsDict,
    SaveClassificationD,
    SaveClassificationd,
    SaveClassificationDict,
    SobelGradientsd,
    SobelGradientsD,
    SobelGradientsDict,
    VoteEnsembleD,
    VoteEnsembled,
    VoteEnsembleDict,
)
from .regularization.array import CutMix, CutOut, MixUp
from .regularization.dictionary import (
    CutMixd,
    CutMixD,
    CutMixDict,
    CutOutd,
    CutOutD,
    CutOutDict,
    MixUpd,
    MixUpD,
    MixUpDict,
)
from .signal.array import (
    SignalContinuousWavelet,
    SignalFillEmpty,
    SignalRandAddGaussianNoise,
    SignalRandAddSine,
    SignalRandAddSinePartial,
    SignalRandAddSquarePulse,
    SignalRandAddSquarePulsePartial,
    SignalRandDrop,
    SignalRandScale,
    SignalRandShift,
    SignalRemoveFrequency,
)
from .signal.dictionary import SignalFillEmptyd, SignalFillEmptyD, SignalFillEmptyDict
from .smooth_field.array import (
    RandSmoothDeform,
    RandSmoothFieldAdjustContrast,
    RandSmoothFieldAdjustIntensity,
    SmoothField,
)
from .smooth_field.dictionary import (
    RandSmoothDeformd,
    RandSmoothDeformD,
    RandSmoothDeformDict,
    RandSmoothFieldAdjustContrastd,
    RandSmoothFieldAdjustContrastD,
    RandSmoothFieldAdjustContrastDict,
    RandSmoothFieldAdjustIntensityd,
    RandSmoothFieldAdjustIntensityD,
    RandSmoothFieldAdjustIntensityDict,
)
from .spatial.array import (
    Affine,
    AffineGrid,
    Flip,
    GridDistortion,
    GridPatch,
    GridSplit,
    Orientation,
    Rand2DElastic,
    Rand3DElastic,
    RandAffine,
    RandAffineGrid,
    RandAxisFlip,
    RandDeformGrid,
    RandFlip,
    RandGridDistortion,
    RandGridPatch,
    RandRotate,
    RandRotate90,
    RandSimulateLowResolution,
    RandZoom,
    Resample,
    ResampleToMatch,
    Resize,
    Rotate,
    Rotate90,
    Spacing,
    SpatialResample,
    Zoom,
)
from .spatial.dictionary import (
    Affined,
    AffineD,
    AffineDict,
    Flipd,
    FlipD,
    FlipDict,
    GridDistortiond,
    GridDistortionD,
    GridDistortionDict,
    GridPatchd,
    GridPatchD,
    GridPatchDict,
    GridSplitd,
    GridSplitD,
    GridSplitDict,
    Orientationd,
    OrientationD,
    OrientationDict,
    Rand2DElasticd,
    Rand2DElasticD,
    Rand2DElasticDict,
    Rand3DElasticd,
    Rand3DElasticD,
    Rand3DElasticDict,
    RandAffined,
    RandAffineD,
    RandAffineDict,
    RandAxisFlipd,
    RandAxisFlipD,
    RandAxisFlipDict,
    RandFlipd,
    RandFlipD,
    RandFlipDict,
    RandGridDistortiond,
    RandGridDistortionD,
    RandGridDistortionDict,
    RandGridPatchd,
    RandGridPatchD,
    RandGridPatchDict,
    RandRotate90d,
    RandRotate90D,
    RandRotate90Dict,
    RandRotated,
    RandRotateD,
    RandRotateDict,
    RandSimulateLowResolutiond,
    RandSimulateLowResolutionD,
    RandSimulateLowResolutionDict,
    RandZoomd,
    RandZoomD,
    RandZoomDict,
    ResampleToMatchd,
    ResampleToMatchD,
    ResampleToMatchDict,
    Resized,
    ResizeD,
    ResizeDict,
    Rotate90d,
    Rotate90D,
    Rotate90Dict,
    Rotated,
    RotateD,
    RotateDict,
    Spacingd,
    SpacingD,
    SpacingDict,
    SpatialResampled,
    SpatialResampleD,
    SpatialResampleDict,
    Zoomd,
    ZoomD,
    ZoomDict,
)
from .spatial.functional import spatial_resample
from .traits import LazyTrait, MultiSampleTrait, RandomizableTrait, ThreadUnsafe
from .transform import LazyTransform, MapTransform, Randomizable, RandomizableTransform, Transform, apply_transform
from .utility.array import (
    AddCoordinateChannels,
    AddExtremePointsChannel,
    AsChannelLast,
    CastToType,
    ClassesToIndices,
    ConvertToMultiChannelBasedOnBratsClasses,
    CuCIM,
    DataStats,
    EnsureChannelFirst,
    EnsureType,
    FgBgToIndices,
    Identity,
    ImageFilter,
    IntensityStats,
    LabelToMask,
    Lambda,
    MapLabelValue,
    RandCuCIM,
    RandIdentity,
    RandImageFilter,
    RandLambda,
    RemoveRepeatedChannel,
    RepeatChannel,
    SimulateDelay,
    SplitDim,
    SqueezeDim,
    ToCupy,
    ToDevice,
    ToNumpy,
    ToPIL,
    TorchVision,
    ToTensor,
    Transpose,
)
from .utility.dictionary import (
    AddCoordinateChannelsd,
    AddCoordinateChannelsD,
    AddCoordinateChannelsDict,
    AddExtremePointsChanneld,
    AddExtremePointsChannelD,
    AddExtremePointsChannelDict,
    AsChannelLastd,
    AsChannelLastD,
    AsChannelLastDict,
    CastToTyped,
    CastToTypeD,
    CastToTypeDict,
    ClassesToIndicesd,
    ClassesToIndicesD,
    ClassesToIndicesDict,
    ConcatItemsd,
    ConcatItemsD,
    ConcatItemsDict,
    ConvertToMultiChannelBasedOnBratsClassesd,
    ConvertToMultiChannelBasedOnBratsClassesD,
    ConvertToMultiChannelBasedOnBratsClassesDict,
    CopyItemsd,
    CopyItemsD,
    CopyItemsDict,
    CuCIMd,
    CuCIMD,
    CuCIMDict,
    DataStatsd,
    DataStatsD,
    DataStatsDict,
    DeleteItemsd,
    DeleteItemsD,
    DeleteItemsDict,
    EnsureChannelFirstd,
    EnsureChannelFirstD,
    EnsureChannelFirstDict,
    EnsureTyped,
    EnsureTypeD,
    EnsureTypeDict,
    FgBgToIndicesd,
    FgBgToIndicesD,
    FgBgToIndicesDict,
    FlattenSubKeysd,
    FlattenSubKeysD,
    FlattenSubKeysDict,
    Identityd,
    IdentityD,
    IdentityDict,
    ImageFilterd,
    ImageFilterD,
    ImageFilterDict,
    IntensityStatsd,
    IntensityStatsD,
    IntensityStatsDict,
    LabelToMaskd,
    LabelToMaskD,
    LabelToMaskDict,
    Lambdad,
    LambdaD,
    LambdaDict,
    MapLabelValued,
    MapLabelValueD,
    MapLabelValueDict,
    RandCuCIMd,
    RandCuCIMD,
    RandCuCIMDict,
    RandImageFilterd,
    RandImageFilterD,
    RandImageFilterDict,
    RandLambdad,
    RandLambdaD,
    RandLambdaDict,
    RandTorchVisiond,
    RandTorchVisionD,
    RandTorchVisionDict,
    RemoveRepeatedChanneld,
    RemoveRepeatedChannelD,
    RemoveRepeatedChannelDict,
    RepeatChanneld,
    RepeatChannelD,
    RepeatChannelDict,
    SelectItemsd,
    SelectItemsD,
    SelectItemsDict,
    SimulateDelayd,
    SimulateDelayD,
    SimulateDelayDict,
    SplitDimd,
    SplitDimD,
    SplitDimDict,
    SqueezeDimd,
    SqueezeDimD,
    SqueezeDimDict,
    ToCupyd,
    ToCupyD,
    ToCupyDict,
    ToDeviced,
    ToDeviceD,
    ToDeviceDict,
    ToNumpyd,
    ToNumpyD,
    ToNumpyDict,
    ToPILd,
    ToPILD,
    ToPILDict,
    TorchVisiond,
    TorchVisionD,
    TorchVisionDict,
    ToTensord,
    ToTensorD,
    ToTensorDict,
    Transposed,
    TransposeD,
    TransposeDict,
)
from .utils import (
    Fourier,
    allow_missing_keys_mode,
    attach_hook,
    check_non_lazy_pending_ops,
    compute_divisible_spatial_size,
    convert_applied_interp_mode,
    convert_pad_mode,
    convert_to_contiguous,
    copypaste_arrays,
    create_control_grid,
    create_grid,
    create_rotate,
    create_scale,
    create_shear,
    create_translate,
    distance_transform_edt,
    equalize_hist,
    extreme_points_to_image,
    generate_label_classes_crop_centers,
    generate_pos_neg_label_crop_centers,
    generate_spatial_bounding_box,
    get_extreme_points,
    get_largest_connected_component_mask,
    get_number_image_type_conversions,
    get_transform_backends,
    img_bounds,
    in_bounds,
    is_empty,
    is_positive,
    map_and_generate_sampling_centers,
    map_binary_to_indices,
    map_classes_to_indices,
    map_spatial_axes,
    print_transform_backends,
    rand_choice,
    remove_small_objects,
    rescale_array,
    rescale_array_int_max,
    rescale_instance_array,
    reset_ops_id,
    resize_center,
    resolves_modes,
    sync_meta_info,
    weighted_patch_samples,
    zero_margins,
)
from .utils_morphological_ops import dilate, erode
from .utils_pytorch_numpy_unification import (
    allclose,
    any_np_pt,
    ascontiguousarray,
    clip,
    concatenate,
    cumsum,
    floor_divide,
    in1d,
    isfinite,
    isnan,
    maximum,
    mode,
    moveaxis,
    nonzero,
    percentile,
    ravel,
    repeat,
    stack,
    unravel_index,
    where,
)
