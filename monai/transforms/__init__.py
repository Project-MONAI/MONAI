# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .adaptors import FunctionSignature, adaptor, apply_alias, to_kwargs
from .compose import Compose
from .croppad.array import (
    BorderPad,
    BoundingRect,
    CenterSpatialCrop,
    CropForeground,
    DivisiblePad,
    RandCropByPosNegLabel,
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
    CenterSpatialCropd,
    CenterSpatialCropD,
    CenterSpatialCropDict,
    CropForegroundd,
    CropForegroundD,
    CropForegroundDict,
    DivisiblePadd,
    DivisiblePadD,
    DivisiblePadDict,
    NumpyPadModeSequence,
    RandCropByPosNegLabeld,
    RandCropByPosNegLabelD,
    RandCropByPosNegLabelDict,
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
from .intensity.array import (
    AdjustContrast,
    DetectEnvelope,
    GaussianSharpen,
    GaussianSmooth,
    GibbsNoise,
    MaskIntensity,
    NormalizeIntensity,
    RandAdjustContrast,
    RandBiasField,
    RandGaussianNoise,
    RandGaussianSharpen,
    RandGaussianSmooth,
    RandGibbsNoise,
    RandHistogramShift,
    RandRicianNoise,
    RandScaleIntensity,
    RandShiftIntensity,
    RandStdShiftIntensity,
    SavitzkyGolaySmooth,
    ScaleIntensity,
    ScaleIntensityRange,
    ScaleIntensityRangePercentiles,
    ShiftIntensity,
    StdShiftIntensity,
    ThresholdIntensity,
)
from .intensity.dictionary import (
    AdjustContrastd,
    AdjustContrastD,
    AdjustContrastDict,
    GaussianSharpend,
    GaussianSharpenD,
    GaussianSharpenDict,
    GaussianSmoothd,
    GaussianSmoothD,
    GaussianSmoothDict,
    GibbsNoised,
    GibbsNoiseD,
    GibbsNoiseDict,
    MaskIntensityd,
    MaskIntensityD,
    MaskIntensityDict,
    NormalizeIntensityd,
    NormalizeIntensityD,
    NormalizeIntensityDict,
    RandAdjustContrastd,
    RandAdjustContrastD,
    RandAdjustContrastDict,
    RandBiasFieldd,
    RandBiasFieldD,
    RandBiasFieldDict,
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
    RandRicianNoised,
    RandRicianNoiseD,
    RandRicianNoiseDict,
    RandScaleIntensityd,
    RandScaleIntensityD,
    RandScaleIntensityDict,
    RandShiftIntensityd,
    RandShiftIntensityD,
    RandShiftIntensityDict,
    RandStdShiftIntensityd,
    RandStdShiftIntensityD,
    RandStdShiftIntensityDict,
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
from .inverse import InvertibleTransform
from .inverse_batch_transform import BatchInverseTransform
from .io.array import LoadImage, SaveImage
from .io.dictionary import LoadImaged, LoadImageD, LoadImageDict, SaveImaged, SaveImageD, SaveImageDict
from .post.array import (
    Activations,
    AsDiscrete,
    KeepLargestConnectedComponent,
    LabelToContour,
    MeanEnsemble,
    ProbNMS,
    VoteEnsemble,
)
from .post.dictionary import (
    Activationsd,
    ActivationsD,
    ActivationsDict,
    AsDiscreted,
    AsDiscreteD,
    AsDiscreteDict,
    Decollated,
    DecollateD,
    DecollateDict,
    Ensembled,
    Invertd,
    InvertD,
    InvertDict,
    KeepLargestConnectedComponentd,
    KeepLargestConnectedComponentD,
    KeepLargestConnectedComponentDict,
    LabelToContourd,
    LabelToContourD,
    LabelToContourDict,
    MeanEnsembled,
    MeanEnsembleD,
    MeanEnsembleDict,
    ProbNMSd,
    ProbNMSD,
    ProbNMSDict,
    VoteEnsembled,
    VoteEnsembleD,
    VoteEnsembleDict,
)
from .spatial.array import (
    Affine,
    AffineGrid,
    Flip,
    Orientation,
    Rand2DElastic,
    Rand3DElastic,
    RandAffine,
    RandAffineGrid,
    RandAxisFlip,
    RandDeformGrid,
    RandFlip,
    RandRotate,
    RandRotate90,
    RandZoom,
    Resample,
    Resize,
    Rotate,
    Rotate90,
    Spacing,
    Zoom,
)
from .spatial.dictionary import (
    Affined,
    AffineD,
    AffineDict,
    Flipd,
    FlipD,
    FlipDict,
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
    RandRotate90d,
    RandRotate90D,
    RandRotate90Dict,
    RandRotated,
    RandRotateD,
    RandRotateDict,
    RandZoomd,
    RandZoomD,
    RandZoomDict,
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
    Zoomd,
    ZoomD,
    ZoomDict,
)
from .transform import MapTransform, Randomizable, RandomizableTransform, Transform, apply_transform
from .utility.array import (
    AddChannel,
    AddExtremePointsChannel,
    AsChannelFirst,
    AsChannelLast,
    CastToType,
    ConvertToMultiChannelBasedOnBratsClasses,
    DataStats,
    EnsureChannelFirst,
    FgBgToIndices,
    Identity,
    LabelToMask,
    Lambda,
    MapLabelValue,
    RemoveRepeatedChannel,
    RepeatChannel,
    SimulateDelay,
    SplitChannel,
    SqueezeDim,
    ToCupy,
    ToNumpy,
    ToPIL,
    TorchVision,
    ToTensor,
    Transpose,
)
from .utility.dictionary import (
    AddChanneld,
    AddChannelD,
    AddChannelDict,
    AddExtremePointsChanneld,
    AddExtremePointsChannelD,
    AddExtremePointsChannelDict,
    AsChannelFirstd,
    AsChannelFirstD,
    AsChannelFirstDict,
    AsChannelLastd,
    AsChannelLastD,
    AsChannelLastDict,
    CastToTyped,
    CastToTypeD,
    CastToTypeDict,
    ConcatItemsd,
    ConcatItemsD,
    ConcatItemsDict,
    ConvertToMultiChannelBasedOnBratsClassesd,
    ConvertToMultiChannelBasedOnBratsClassesD,
    ConvertToMultiChannelBasedOnBratsClassesDict,
    CopyItemsd,
    CopyItemsD,
    CopyItemsDict,
    DataStatsd,
    DataStatsD,
    DataStatsDict,
    DeleteItemsd,
    DeleteItemsD,
    DeleteItemsDict,
    EnsureChannelFirstd,
    EnsureChannelFirstD,
    EnsureChannelFirstDict,
    FgBgToIndicesd,
    FgBgToIndicesD,
    FgBgToIndicesDict,
    Identityd,
    IdentityD,
    IdentityDict,
    LabelToMaskd,
    LabelToMaskD,
    LabelToMaskDict,
    Lambdad,
    LambdaD,
    LambdaDict,
    MapLabelValued,
    MapLabelValueD,
    MapLabelValueDict,
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
    SplitChanneld,
    SplitChannelD,
    SplitChannelDict,
    SqueezeDimd,
    SqueezeDimD,
    SqueezeDimDict,
    ToCupyd,
    ToCupyD,
    ToCupyDict,
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
    allow_missing_keys_mode,
    compute_divisible_spatial_size,
    convert_inverse_interp_mode,
    copypaste_arrays,
    create_control_grid,
    create_grid,
    create_rotate,
    create_scale,
    create_shear,
    create_translate,
    extreme_points_to_image,
    generate_pos_neg_label_crop_centers,
    generate_spatial_bounding_box,
    get_extreme_points,
    get_largest_connected_component_mask,
    img_bounds,
    in_bounds,
    is_empty,
    is_positive,
    map_binary_to_indices,
    map_spatial_axes,
    rand_choice,
    rescale_array,
    rescale_array_int_max,
    rescale_instance_array,
    resize_center,
    weighted_patch_samples,
    zero_margins,
)
