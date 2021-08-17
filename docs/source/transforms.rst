:github_url: https://github.com/Project-MONAI/MONAI

.. _transform_api:

Transforms
==========

Generic Interfaces
------------------
.. automodule:: monai.transforms
.. currentmodule:: monai.transforms

`Transform`
^^^^^^^^^^^
.. autoclass:: Transform
    :members:
    :special-members: __call__

`MapTransform`
^^^^^^^^^^^^^^
.. autoclass:: MapTransform
    :members:
    :special-members: __call__

`Randomizable`
^^^^^^^^^^^^^^
.. autoclass:: Randomizable
    :members:

`RandomizableTransform`
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: RandomizableTransform
    :members:

`Compose`
^^^^^^^^^
.. autoclass:: Compose
    :members:
    :special-members: __call__

`InvertibleTransform`
^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: InvertibleTransform
    :members:

`BatchInverseTransform`
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: BatchInverseTransform
    :members:

`Decollated`
^^^^^^^^^^^^
.. autoclass:: Decollated
    :members:

`Fourier`
^^^^^^^^^^^^^
.. autoclass:: Fourier
    :members:

Vanilla Transforms
------------------

Crop and Pad
^^^^^^^^^^^^

`SpatialPad`
""""""""""""
.. autoclass:: SpatialPad
    :members:
    :special-members: __call__

`BorderPad`
"""""""""""
.. autoclass:: BorderPad
    :members:
    :special-members: __call__

`DivisiblePad`
""""""""""""""
.. autoclass:: DivisiblePad
    :members:
    :special-members: __call__

`SpatialCrop`
"""""""""""""
.. autoclass:: SpatialCrop
    :members:
    :special-members: __call__

`CenterSpatialCrop`
"""""""""""""""""""
.. autoclass:: CenterSpatialCrop
    :members:
    :special-members: __call__

`RandSpatialCrop`
"""""""""""""""""
.. autoclass:: RandSpatialCrop
    :members:
    :special-members: __call__

`RandSpatialCropSamples`
""""""""""""""""""""""""
.. autoclass:: RandSpatialCropSamples
    :members:
    :special-members: __call__

`CropForeground`
""""""""""""""""
.. autoclass:: CropForeground
    :members:
    :special-members: __call__

`RandWeightedCrop`
""""""""""""""""""
.. autoclass:: RandWeightedCrop
    :members:
    :special-members: __call__

`RandCropByPosNegLabel`
"""""""""""""""""""""""
.. autoclass:: RandCropByPosNegLabel
    :members:
    :special-members: __call__

`RandCropByLabelClasses`
""""""""""""""""""""""""
.. autoclass:: RandCropByLabelClasses
    :members:
    :special-members: __call__

`ResizeWithPadOrCrop`
"""""""""""""""""""""
.. autoclass:: ResizeWithPadOrCrop
    :members:
    :special-members: __call__

`BoundingRect`
""""""""""""""
.. autoclass:: BoundingRect
    :members:
    :special-members: __call__

`RandScaleCrop`
"""""""""""""""
.. autoclass:: RandScaleCrop
    :members:
    :special-members: __call__

`CenterScaleCrop`
"""""""""""""""""
.. autoclass:: CenterScaleCrop
    :members:
    :special-members: __call__

Intensity
^^^^^^^^^

`RandGaussianNoise`
"""""""""""""""""""
.. autoclass:: RandGaussianNoise
    :members:
    :special-members: __call__

`ShiftIntensity`
""""""""""""""""
.. autoclass:: ShiftIntensity
    :members:
    :special-members: __call__

`RandShiftIntensity`
""""""""""""""""""""
.. autoclass:: RandShiftIntensity
    :members:
    :special-members: __call__

`StdShiftIntensity`
"""""""""""""""""""
.. autoclass:: StdShiftIntensity
    :members:
    :special-members: __call__

`RandStdShiftIntensity`
"""""""""""""""""""""""
.. autoclass:: RandStdShiftIntensity
    :members:
    :special-members: __call__

`RandBiasField`
"""""""""""""""
.. autoclass:: RandBiasField
    :members:
    :special-members: __call__

`ScaleIntensity`
""""""""""""""""
.. autoclass:: ScaleIntensity
    :members:
    :special-members: __call__

`RandScaleIntensity`
""""""""""""""""""""
.. autoclass:: RandScaleIntensity
    :members:
    :special-members: __call__

`NormalizeIntensity`
""""""""""""""""""""
.. autoclass:: NormalizeIntensity
    :members:
    :special-members: __call__

`ThresholdIntensity`
""""""""""""""""""""
.. autoclass:: ThresholdIntensity
    :members:
    :special-members: __call__

`ScaleIntensityRange`
"""""""""""""""""""""
.. autoclass:: ScaleIntensityRange
    :members:
    :special-members: __call__

`ScaleIntensityRangePercentiles`
""""""""""""""""""""""""""""""""
.. autoclass:: ScaleIntensityRangePercentiles
    :members:
    :special-members: __call__

`AdjustContrast`
""""""""""""""""
.. autoclass:: AdjustContrast
    :members:
    :special-members: __call__

`RandAdjustContrast`
""""""""""""""""""""
.. autoclass:: RandAdjustContrast
    :members:
    :special-members: __call__

`MaskIntensity`
"""""""""""""""
.. autoclass:: MaskIntensity
    :members:
    :special-members: __call__

`SavitzkyGolaySmooth`
"""""""""""""""""""""
.. autoclass:: SavitzkyGolaySmooth
    :members:
    :special-members: __call__

`GaussianSmooth`
""""""""""""""""
.. autoclass:: GaussianSmooth
    :members:
    :special-members: __call__

`RandGaussianSmooth`
""""""""""""""""""""
.. autoclass:: RandGaussianSmooth
    :members:
    :special-members: __call__

`GaussianSharpen`
"""""""""""""""""
.. autoclass:: GaussianSharpen
    :members:
    :special-members: __call__

`RandGaussianSharpen`
"""""""""""""""""""""
.. autoclass:: RandGaussianSharpen
    :members:
    :special-members: __call__

`RandHistogramShift`
""""""""""""""""""""
.. autoclass:: RandHistogramShift
    :members:
    :special-members: __call__

`DetectEnvelope`
""""""""""""""""
.. autoclass:: DetectEnvelope
    :members:
    :special-members: __call__

`GibbsNoise`
""""""""""""
.. autoclass:: GibbsNoise
    :members:
    :special-members: __call__

`RandGibbsNoise`
""""""""""""""""
.. autoclass:: RandGibbsNoise
    :members:
    :special-members: __call__

`KSpaceSpikeNoise`
""""""""""""""""""
.. autoclass:: KSpaceSpikeNoise
    :members:
    :special-members: __call__

`RandKSpaceSpikeNoise`
""""""""""""""""""""""
 .. autoclass:: RandKSpaceSpikeNoise
     :members:
     :special-members: __call__

`RandCoarseDropout`
"""""""""""""""""""
 .. autoclass:: RandCoarseDropout
     :members:
     :special-members: __call__

`HistogramNormalize`
""""""""""""""""""""
 .. autoclass:: HistogramNormalize
     :members:
     :special-members: __call__


IO
^^

`LoadImage`
"""""""""""
.. autoclass:: LoadImage
    :members:
    :special-members: __call__

`SaveImage`
"""""""""""
.. autoclass:: SaveImage
    :members:
    :special-members: __call__


NVIDIA Tool Extension (NVTX)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`RangePush`
"""""""""""
.. autoclass:: RangePush

`RandRangePush`
"""""""""""""""
.. autoclass:: RandRangePush

`RangePop`
""""""""""
.. autoclass:: RangePop

`RandRangePop`
""""""""""""""
.. autoclass:: RandRangePop

`Range`
"""""""
.. autoclass:: Range

`RandRange`
"""""""""""
.. autoclass:: RandRange

`Mark`
""""""
.. autoclass:: Mark

`RandMark`
""""""""""
.. autoclass:: RandMark


Post-processing
^^^^^^^^^^^^^^^

`Activations`
"""""""""""""
.. autoclass:: Activations
    :members:
    :special-members: __call__

`AsDiscrete`
""""""""""""
.. autoclass:: AsDiscrete
    :members:
    :special-members: __call__

`KeepLargestConnectedComponent`
"""""""""""""""""""""""""""""""
.. autoclass:: KeepLargestConnectedComponent
    :members:
    :special-members: __call__

`LabelFilter`
"""""""""""""
.. autoclass:: LabelFilter
    :members:
    :special-members: __call__

`FillHoles`
"""""""""""
.. autoclass:: FillHoles
    :members:
    :special-members: __call__

`LabelToContour`
""""""""""""""""
.. autoclass:: LabelToContour
    :members:
    :special-members: __call__

`MeanEnsemble`
""""""""""""""
.. autoclass:: MeanEnsemble
    :members:
    :special-members: __call__

`Prob NMS`
""""""""""
.. autoclass:: ProbNMS
  :members:

`VoteEnsemble`
""""""""""""""
.. autoclass:: VoteEnsemble
    :members:
    :special-members: __call__

Spatial
^^^^^^^

`Spacing`
"""""""""
.. autoclass:: Spacing
    :members:
    :special-members: __call__

`Orientation`
"""""""""""""
.. autoclass:: Orientation
    :members:
    :special-members: __call__

`RandRotate`
""""""""""""
.. autoclass:: RandRotate
    :members:
    :special-members: __call__

`RandFlip`
""""""""""
.. autoclass:: RandFlip
    :members:
    :special-members: __call__

`RandAxisFlip`
""""""""""""""
.. autoclass:: RandAxisFlip
    :members:
    :special-members: __call__

`RandZoom`
""""""""""
.. autoclass:: RandZoom
    :members:
    :special-members: __call__

`Affine`
""""""""
.. autoclass:: Affine
    :members:
    :special-members: __call__

`Resample`
""""""""""
.. autoclass:: Resample
    :members:
    :special-members: __call__

`RandAffine`
""""""""""""
.. autoclass:: RandAffine
    :members:
    :special-members: __call__

`RandDeformGrid`
""""""""""""""""
.. autoclass:: RandDeformGrid
    :members:
    :special-members: __call__

`AffineGrid`
""""""""""""""""
.. autoclass:: AffineGrid
    :members:
    :special-members: __call__

`RandAffineGrid`
""""""""""""""""
.. autoclass:: RandAffineGrid
    :members:
    :special-members: __call__

`Rand2DElastic`
"""""""""""""""
.. autoclass:: Rand2DElastic
    :members:
    :special-members: __call__

`Rand3DElastic`
"""""""""""""""
.. autoclass:: Rand3DElastic
    :members:
    :special-members: __call__

`Rotate90`
""""""""""
.. autoclass:: Rotate90
    :members:
    :special-members: __call__

`RandRotate90`
""""""""""""""
.. autoclass:: RandRotate90
    :members:
    :special-members: __call__

`Flip`
""""""
.. autoclass:: Flip
    :members:
    :special-members: __call__

`Resize`
""""""""
.. autoclass:: Resize
    :members:
    :special-members: __call__

`Rotate`
""""""""
.. autoclass:: Rotate
    :members:
    :special-members: __call__

`Zoom`
""""""
.. autoclass:: Zoom
    :members:
    :special-members: __call__

`AddCoordinateChannels`
"""""""""""""""""""""""
.. autoclass:: AddCoordinateChannels
    :members:
    :special-members: __call__

Utility
^^^^^^^

`Identity`
""""""""""
.. autoclass:: Identity
    :members:
    :special-members: __call__

`AsChannelFirst`
""""""""""""""""
.. autoclass:: AsChannelFirst
    :members:
    :special-members: __call__

`AsChannelLast`
"""""""""""""""
.. autoclass:: AsChannelLast
    :members:
    :special-members: __call__

`AddChannel`
""""""""""""
.. autoclass:: AddChannel
    :members:
    :special-members: __call__

`EnsureChannelFirst`
""""""""""""""""""""
.. autoclass:: EnsureChannelFirst
    :members:
    :special-members: __call__

`RepeatChannel`
"""""""""""""""
.. autoclass:: RepeatChannel
    :members:
    :special-members: __call__

`SplitChannel`
""""""""""""""
.. autoclass:: SplitChannel
    :members:
    :special-members: __call__

`CastToType`
""""""""""""
.. autoclass:: CastToType
    :members:
    :special-members: __call__

`ToTensor`
""""""""""
.. autoclass:: ToTensor
    :members:
    :special-members: __call__

`ToNumpy`
"""""""""
.. autoclass:: ToNumpy
    :members:
    :special-members: __call__

`ToCupy`
""""""""
.. autoclass:: ToCupy
    :members:
    :special-members: __call__


`Transpose`
"""""""""""
.. autoclass:: Transpose
    :members:
    :special-members: __call__

`SqueezeDim`
""""""""""""
.. autoclass:: SqueezeDim
    :members:
    :special-members: __call__

`DataStats`
"""""""""""
.. autoclass:: DataStats
    :members:
    :special-members: __call__

`SimulateDelay`
"""""""""""""""
.. autoclass:: SimulateDelay
    :members:
    :special-members: __call__

`Lambda`
""""""""
.. autoclass:: Lambda
    :members:
    :special-members: __call__

`RandLambda`
""""""""""""
.. autoclass:: RandLambda
    :members:
    :special-members: __call__

`LabelToMask`
"""""""""""""
.. autoclass:: LabelToMask
    :members:
    :special-members: __call__

`FgBgToIndices`
"""""""""""""""
.. autoclass:: FgBgToIndices
    :members:
    :special-members: __call__

`ClassesToIndices`
""""""""""""""""""
.. autoclass:: ClassesToIndices
    :members:
    :special-members: __call__

`ConvertToMultiChannelBasedOnBratsClasses`
""""""""""""""""""""""""""""""""""""""""""
.. autoclass:: ConvertToMultiChannelBasedOnBratsClasses
    :members:
    :special-members: __call__

`AddExtremePointsChannel`
"""""""""""""""""""""""""
.. autoclass:: AddExtremePointsChannel
    :members:
    :special-members: __call__

`TorchVision`
"""""""""""""
.. autoclass:: TorchVision
    :members:
    :special-members: __call__

`MapLabelValue`
"""""""""""""""
.. autoclass:: MapLabelValue
    :members:
    :special-members: __call__

`EnsureType`
""""""""""""
.. autoclass:: EnsureType
    :members:
    :special-members: __call__

`IntensityStats`
""""""""""""""""
 .. autoclass:: IntensityStats
     :members:
     :special-members: __call__

`ToDevice`
""""""""""
 .. autoclass:: ToDevice
     :members:
     :special-members: __call__


Dictionary Transforms
---------------------

Crop and Pad (Dict)
^^^^^^^^^^^^^^^^^^^

`SpatialPadd`
"""""""""""""
.. autoclass:: SpatialPadd
    :members:
    :special-members: __call__

`BorderPadd`
""""""""""""
.. autoclass:: BorderPadd
    :members:
    :special-members: __call__

`DivisiblePadd`
"""""""""""""""
.. autoclass:: DivisiblePadd
    :members:
    :special-members: __call__

`SpatialCropd`
""""""""""""""
.. autoclass:: SpatialCropd
    :members:
    :special-members: __call__

`CenterSpatialCropd`
""""""""""""""""""""
.. autoclass:: CenterSpatialCropd
    :members:
    :special-members: __call__

`RandSpatialCropd`
""""""""""""""""""
.. autoclass:: RandSpatialCropd
    :members:
    :special-members: __call__

`RandSpatialCropSamplesd`
"""""""""""""""""""""""""
.. autoclass:: RandSpatialCropSamplesd
    :members:
    :special-members: __call__

`CropForegroundd`
"""""""""""""""""
.. autoclass:: CropForegroundd
    :members:
    :special-members: __call__

`RandWeightedCropd`
"""""""""""""""""""
.. autoclass:: RandWeightedCropd
    :members:
    :special-members: __call__

`RandCropByPosNegLabeld`
""""""""""""""""""""""""
.. autoclass:: RandCropByPosNegLabeld
    :members:
    :special-members: __call__

`RandCropByLabelClassesd`
"""""""""""""""""""""""""
.. autoclass:: RandCropByLabelClassesd
    :members:
    :special-members: __call__

`ResizeWithPadOrCropd`
""""""""""""""""""""""
.. autoclass:: ResizeWithPadOrCropd
    :members:
    :special-members: __call__

`BoundingRectd`
"""""""""""""""
.. autoclass:: BoundingRectd
    :members:
    :special-members: __call__

`RandScaleCropd`
""""""""""""""""
.. autoclass:: RandScaleCropd
    :members:
    :special-members: __call__

`CenterScaleCropd`
""""""""""""""""""
.. autoclass:: CenterScaleCropd
    :members:
    :special-members: __call__

Intensity (Dict)
^^^^^^^^^^^^^^^^

`RandGaussianNoised`
""""""""""""""""""""
.. autoclass:: RandGaussianNoised
    :members:
    :special-members: __call__

`ShiftIntensityd`
"""""""""""""""""
.. autoclass:: ShiftIntensityd
    :members:
    :special-members: __call__

`RandShiftIntensityd`
"""""""""""""""""""""
.. autoclass:: RandShiftIntensityd
    :members:
    :special-members: __call__

`StdShiftIntensityd`
""""""""""""""""""""
.. autoclass:: StdShiftIntensityd
    :members:
    :special-members: __call__

`RandStdShiftIntensityd`
""""""""""""""""""""""""
.. autoclass:: RandStdShiftIntensityd
    :members:
    :special-members: __call__

`RandBiasFieldd`
""""""""""""""""
.. autoclass:: RandBiasFieldd
    :members:
    :special-members: __call__

`ScaleIntensityd`
"""""""""""""""""
.. autoclass:: ScaleIntensityd
    :members:
    :special-members: __call__

`RandScaleIntensityd`
"""""""""""""""""""""
.. autoclass:: RandScaleIntensityd
    :members:
    :special-members: __call__

`NormalizeIntensityd`
"""""""""""""""""""""
.. autoclass:: NormalizeIntensityd
    :members:
    :special-members: __call__

`ThresholdIntensityd`
"""""""""""""""""""""
.. autoclass:: ThresholdIntensityd
    :members:
    :special-members: __call__

`ScaleIntensityRanged`
""""""""""""""""""""""
.. autoclass:: ScaleIntensityRanged
    :members:
    :special-members: __call__

`GibbsNoised`
""""""""""""""
.. autoclass:: GibbsNoised
    :members:
    :special-members: __call__

`RandGibbsNoised`
""""""""""""""""""
.. autoclass:: RandGibbsNoised
    :members:
    :special-members: __call__

`KSpaceSpikeNoised`
""""""""""""""""""""""
.. autoclass:: KSpaceSpikeNoised
    :members:
    :special-members: __call__

`RandKSpaceSpikeNoised`
"""""""""""""""""""""""""
.. autoclass:: RandKSpaceSpikeNoised
    :members:
    :special-members: __call__

`ScaleIntensityRangePercentilesd`
"""""""""""""""""""""""""""""""""
.. autoclass:: ScaleIntensityRangePercentilesd
    :members:
    :special-members: __call__

`AdjustContrastd`
"""""""""""""""""
.. autoclass:: AdjustContrastd
    :members:
    :special-members: __call__

`RandAdjustContrastd`
"""""""""""""""""""""
.. autoclass:: RandAdjustContrastd
    :members:
    :special-members: __call__

`MaskIntensityd`
""""""""""""""""
.. autoclass:: MaskIntensityd
    :members:
    :special-members: __call__

`GaussianSmoothd`
"""""""""""""""""
.. autoclass:: GaussianSmoothd
    :members:
    :special-members: __call__

`RandGaussianSmoothd`
"""""""""""""""""""""
.. autoclass:: RandGaussianSmoothd
    :members:
    :special-members: __call__

`GaussianSharpend`
""""""""""""""""""
.. autoclass:: GaussianSharpend
    :members:
    :special-members: __call__

`RandGaussianSharpend`
""""""""""""""""""""""
.. autoclass:: RandGaussianSharpend
    :members:
    :special-members: __call__

`RandHistogramShiftd`
"""""""""""""""""""""
.. autoclass:: RandHistogramShiftd
    :members:
    :special-members: __call__

`RandCoarseDropoutd`
""""""""""""""""""""
.. autoclass:: RandCoarseDropoutd
    :members:
    :special-members: __call__

`HistogramNormalized`
"""""""""""""""""""""
 .. autoclass:: HistogramNormalized
     :members:
     :special-members: __call__


IO (Dict)
^^^^^^^^^

`LoadImaged`
""""""""""""
.. autoclass:: LoadImaged
    :members:
    :special-members: __call__

`SaveImaged`
""""""""""""
.. autoclass:: SaveImaged
    :members:
    :special-members: __call__

Post-processing (Dict)
^^^^^^^^^^^^^^^^^^^^^^

`Activationsd`
""""""""""""""
.. autoclass:: Activationsd
    :members:
    :special-members: __call__

`AsDiscreted`
"""""""""""""
.. autoclass:: AsDiscreted
    :members:
    :special-members: __call__

`KeepLargestConnectedComponentd`
""""""""""""""""""""""""""""""""
.. autoclass:: KeepLargestConnectedComponentd
    :members:
    :special-members: __call__

`LabelFilterd`
""""""""""""""
.. autoclass:: LabelFilterd
    :members:
    :special-members: __call__

`FillHolesd`
""""""""""""
.. autoclass:: FillHolesd
    :members:
    :special-members: __call__

`LabelToContourd`
"""""""""""""""""
.. autoclass:: LabelToContourd
    :members:
    :special-members: __call__

`Ensembled`
"""""""""""
.. autoclass:: Ensembled
    :members:
    :special-members: __call__

`MeanEnsembled`
"""""""""""""""
.. autoclass:: MeanEnsembled
    :members:
    :special-members: __call__

`VoteEnsembled`
"""""""""""""""
.. autoclass:: VoteEnsembled
    :members:
    :special-members: __call__

`Invertd`
"""""""""
.. autoclass:: Invertd
    :members:
    :special-members: __call__

`SaveClassificationd`
"""""""""""""""""""""
.. autoclass:: SaveClassificationd
    :members:
    :special-members: __call__

Spatial (Dict)
^^^^^^^^^^^^^^

`Spacingd`
""""""""""
.. autoclass:: Spacingd
    :members:
    :special-members: __call__

`Orientationd`
""""""""""""""
.. autoclass:: Orientationd
    :members:
    :special-members: __call__

`Flipd`
"""""""
.. autoclass:: Flipd
    :members:
    :special-members: __call__

`RandFlipd`
"""""""""""
.. autoclass:: RandFlipd
    :members:
    :special-members: __call__

`RandAxisFlipd`
"""""""""""""""
.. autoclass:: RandAxisFlipd
    :members:
    :special-members: __call__

`Rotated`
"""""""""
.. autoclass:: Rotated
    :members:
    :special-members: __call__

`RandRotated`
"""""""""""""
.. autoclass:: RandRotated
    :members:
    :special-members: __call__

`Zoomd`
"""""""
.. autoclass:: Zoomd
    :members:
    :special-members: __call__

`RandZoomd`
"""""""""""
.. autoclass:: RandZoomd
    :members:
    :special-members: __call__

`RandRotate90d`
"""""""""""""""
.. autoclass:: RandRotate90d
    :members:
    :special-members: __call__

`Rotate90d`
"""""""""""
.. autoclass:: Rotate90d
    :members:
    :special-members: __call__

`Resized`
"""""""""
.. autoclass:: Resized
    :members:
    :special-members: __call__

`Affined`
"""""""""
.. autoclass:: Affined
    :members:
    :special-members: __call__

`RandAffined`
"""""""""""""
.. autoclass:: RandAffined
    :members:
    :special-members: __call__

`Rand2DElasticd`
""""""""""""""""
.. autoclass:: Rand2DElasticd
    :members:
    :special-members: __call__

`Rand3DElasticd`
""""""""""""""""
.. autoclass:: Rand3DElasticd
    :members:
    :special-members: __call__

`AddCoordinateChannelsd`
""""""""""""""""""""""""
.. autoclass:: AddCoordinateChannelsd
    :members:
    :special-members: __call__

Utility (Dict)
^^^^^^^^^^^^^^

`Identityd`
"""""""""""
.. autoclass:: Identityd
    :members:
    :special-members: __call__

`AsChannelFirstd`
"""""""""""""""""
.. autoclass:: AsChannelFirstd
    :members:
    :special-members: __call__

`AsChannelLastd`
""""""""""""""""
.. autoclass:: AsChannelLastd
    :members:
    :special-members: __call__

`AddChanneld`
"""""""""""""
.. autoclass:: AddChanneld
    :members:
    :special-members: __call__

`EnsureChannelFirstd`
"""""""""""""""""""""
.. autoclass:: EnsureChannelFirstd
    :members:
    :special-members: __call__

`RepeatChanneld`
""""""""""""""""
.. autoclass:: RepeatChanneld
    :members:
    :special-members: __call__

`SplitChanneld`
"""""""""""""""
.. autoclass:: SplitChanneld
    :members:
    :special-members: __call__

`CastToTyped`
"""""""""""""
.. autoclass:: CastToTyped
    :members:
    :special-members: __call__

`ToTensord`
"""""""""""
.. autoclass:: ToTensord
    :members:
    :special-members: __call__

`ToNumpyd`
""""""""""
.. autoclass:: ToNumpyd
    :members:
    :special-members: __call__

`ToCupyd`
"""""""""
.. autoclass:: ToCupyd
    :members:
    :special-members: __call__

`DeleteItemsd`
""""""""""""""
.. autoclass:: DeleteItemsd
    :members:
    :special-members: __call__

`SelectItemsd`
""""""""""""""
.. autoclass:: SelectItemsd
    :members:
    :special-members: __call__

`SqueezeDimd`
"""""""""""""
.. autoclass:: SqueezeDimd
    :members:
    :special-members: __call__

`DataStatsd`
""""""""""""
.. autoclass:: DataStatsd
    :members:
    :special-members: __call__

`SimulateDelayd`
""""""""""""""""
.. autoclass:: SimulateDelayd
    :members:
    :special-members: __call__

`CopyItemsd`
""""""""""""
.. autoclass:: CopyItemsd
    :members:
    :special-members: __call__

`ConcatItemsd`
""""""""""""""
.. autoclass:: ConcatItemsd
    :members:
    :special-members: __call__

`Lambdad`
"""""""""
.. autoclass:: Lambdad
    :members:
    :special-members: __call__

`RandLambdad`
"""""""""""""
.. autoclass:: RandLambdad
    :members:
    :special-members: __call__

`LabelToMaskd`
""""""""""""""
.. autoclass:: LabelToMaskd
    :members:
    :special-members: __call__

`FgBgToIndicesd`
""""""""""""""""
.. autoclass:: FgBgToIndicesd
    :members:
    :special-members: __call__

`ClassesToIndicesd`
"""""""""""""""""""
.. autoclass:: ClassesToIndicesd
    :members:
    :special-members: __call__

`ConvertToMultiChannelBasedOnBratsClassesd`
"""""""""""""""""""""""""""""""""""""""""""
.. autoclass:: ConvertToMultiChannelBasedOnBratsClassesd
    :members:
    :special-members: __call__

`AddExtremePointsChanneld`
""""""""""""""""""""""""""
.. autoclass:: AddExtremePointsChanneld
    :members:
    :special-members: __call__

`TorchVisiond`
""""""""""""""
.. autoclass:: TorchVisiond
    :members:
    :special-members: __call__

`RandTorchVisiond`
""""""""""""""""""
.. autoclass:: RandTorchVisiond
    :members:
    :special-members: __call__

`MapLabelValued`
""""""""""""""""
.. autoclass:: MapLabelValued
    :members:
    :special-members: __call__

`EnsureTyped`
"""""""""""""
.. autoclass:: EnsureTyped
    :members:
    :special-members: __call__

`IntensityStatsd`
"""""""""""""""""
.. autoclass:: IntensityStatsd
    :members:
    :special-members: __call__

`ToDeviced`
"""""""""""
 .. autoclass:: ToDeviced
     :members:
     :special-members: __call__


Transform Adaptors
------------------
.. automodule:: monai.transforms.adaptors

`adaptor`
^^^^^^^^^
.. autofunction:: monai.transforms.adaptors.adaptor

`apply_alias`
^^^^^^^^^^^^^
.. autofunction:: monai.transforms.adaptors.apply_alias

`to_kwargs`
^^^^^^^^^^^
.. autofunction:: monai.transforms.adaptors.to_kwargs

Utilities
---------
.. automodule:: monai.transforms.utils
    :members:
