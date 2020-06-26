:github_url: https://github.com/Project-MONAI/MONAI

.. _transform_api:

Transforms
==========

Generic Interfaces
------------------
.. automodule:: monai.transforms
.. currentmodule:: monai.transforms

`Transform`
~~~~~~~~~~~
.. autoclass:: Transform
    :members:
    :special-members: __call__

`MapTransform`
~~~~~~~~~~~~~~
.. autoclass:: MapTransform
    :members:
    :special-members: __call__

`Randomizable`
~~~~~~~~~~~~~~
.. autoclass:: Randomizable
    :members:

`Compose`
~~~~~~~~~
.. autoclass:: Compose
    :members:
    :special-members: __call__


Vanilla Transforms
------------------

`Spacing`
~~~~~~~~~
.. autoclass:: Spacing
    :members:
    :special-members: __call__

`Orientation`
~~~~~~~~~~~~~
.. autoclass:: Orientation
    :members:
    :special-members: __call__

`LoadNifti`
~~~~~~~~~~~
.. autoclass:: LoadNifti
    :members:
    :special-members: __call__

`LoadPNG`
~~~~~~~~~
.. autoclass:: LoadPNG
    :members:
    :special-members: __call__

`Identity`
~~~~~~~~~~
.. autoclass:: Identity
    :members:
    :special-members: __call__

`AsChannelFirst`
~~~~~~~~~~~~~~~~
.. autoclass:: AsChannelFirst
    :members:
    :special-members: __call__

`AsChannelLast`
~~~~~~~~~~~~~~~
.. autoclass:: AsChannelLast
    :members:
    :special-members: __call__

`AddChannel`
~~~~~~~~~~~~
.. autoclass:: AddChannel
    :members:
    :special-members: __call__

`RepeatChannel`
~~~~~~~~~~~~~~~
.. autoclass:: RepeatChannel
    :members:
    :special-members: __call__

`CastToType`
~~~~~~~~~~~~
.. autoclass:: CastToType
    :members:
    :special-members: __call__

`ToTensor`
~~~~~~~~~~
.. autoclass:: ToTensor
    :members:
    :special-members: __call__

`ToNumpy`
~~~~~~~~~
.. autoclass:: ToNumpy
    :members:
    :special-members: __call__

`Transpose`
~~~~~~~~~~~
.. autoclass:: Transpose
    :members:
    :special-members: __call__

`RandGaussianNoise`
~~~~~~~~~~~~~~~~~~~
.. autoclass:: RandGaussianNoise
    :members:
    :special-members: __call__

`Flip`
~~~~~~
.. autoclass:: Flip
    :members:
    :special-members: __call__

`Resize`
~~~~~~~~
.. autoclass:: Resize
    :members:
    :special-members: __call__

`Rotate`
~~~~~~~~
.. autoclass:: Rotate
    :members:
    :special-members: __call__

`Zoom`
~~~~~~
.. autoclass:: Zoom
    :members:
    :special-members: __call__

`ShiftIntensity`
~~~~~~~~~~~~~~~~
.. autoclass:: ShiftIntensity
    :members:
    :special-members: __call__

`RandShiftIntensity`
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: RandShiftIntensity
    :members:
    :special-members: __call__

`ScaleIntensity`
~~~~~~~~~~~~~~~~
.. autoclass:: ScaleIntensity
    :members:
    :special-members: __call__

`RandScaleIntensity`
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: RandScaleIntensity
    :members:
    :special-members: __call__

`NormalizeIntensity`
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: NormalizeIntensity
    :members:
    :special-members: __call__

`ThresholdIntensity`
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ThresholdIntensity
    :members:
    :special-members: __call__

`ScaleIntensityRange`
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ScaleIntensityRange
    :members:
    :special-members: __call__

`AdjustContrast`
~~~~~~~~~~~~~~~~
.. autoclass:: AdjustContrast
    :members:
    :special-members: __call__

`RandAdjustContrast`
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: RandAdjustContrast
    :members:
    :special-members: __call__

`Rotate90`
~~~~~~~~~~
.. autoclass:: Rotate90
    :members:
    :special-members: __call__

`RandRotate90`
~~~~~~~~~~~~~~
.. autoclass:: RandRotate90
    :members:
    :special-members: __call__

`SpatialPad`
~~~~~~~~~~~~
.. autoclass:: SpatialPad
    :members:
    :special-members: __call__

`BorderPad`
~~~~~~~~~~~
.. autoclass:: BorderPad
    :members:
    :special-members: __call__

`DivisiblePad`
~~~~~~~~~~~~~~
.. autoclass:: DivisiblePad
    :members:
    :special-members: __call__

`SpatialCrop`
~~~~~~~~~~~~~
.. autoclass:: SpatialCrop
    :members:
    :special-members: __call__

`CenterSpatialCrop`
~~~~~~~~~~~~~~~~~~~
.. autoclass:: CenterSpatialCrop
    :members:
    :special-members: __call__

`RandSpatialCrop`
~~~~~~~~~~~~~~~~~
.. autoclass:: RandSpatialCrop
    :members:
    :special-members: __call__

`RandSpatialCropSamples`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: RandSpatialCropSamples
    :members:
    :special-members: __call__

`CropForeground`
~~~~~~~~~~~~~~~~
.. autoclass:: CropForeground
    :members:
    :special-members: __call__

`RandRotate`
~~~~~~~~~~~~
.. autoclass:: RandRotate
    :members:
    :special-members: __call__

`RandFlip`
~~~~~~~~~~
.. autoclass:: RandFlip
    :members:
    :special-members: __call__

`RandZoom`
~~~~~~~~~~
.. autoclass:: RandZoom
    :members:
    :special-members: __call__

`Affine`
~~~~~~~~
.. autoclass:: Affine
    :members:
    :special-members: __call__

`Resample`
~~~~~~~~~~
.. autoclass:: Resample
    :members:
    :special-members: __call__

`RandAffine`
~~~~~~~~~~~~
.. autoclass:: RandAffine
    :members:
    :special-members: __call__

`RandDeformGrid`
~~~~~~~~~~~~~~~~
.. autoclass:: RandDeformGrid
    :members:
    :special-members: __call__

`RandAffineGrid`
~~~~~~~~~~~~~~~~
.. autoclass:: RandAffineGrid
    :members:
    :special-members: __call__

`Rand2DElastic`
~~~~~~~~~~~~~~~
.. autoclass:: Rand2DElastic
    :members:
    :special-members: __call__

`Rand3DElastic`
~~~~~~~~~~~~~~~
.. autoclass:: Rand3DElastic
    :members:
    :special-members: __call__

`SqueezeDim`
~~~~~~~~~~~~
.. autoclass:: SqueezeDim
    :members:
    :special-members: __call__

`DataStats`
~~~~~~~~~~~
.. autoclass:: DataStats
    :members:
    :special-members: __call__

`SimulateDelay`
~~~~~~~~~~~~~~~
.. autoclass:: SimulateDelay
    :members:
    :special-members: __call__

`SplitChannel`
~~~~~~~~~~~~~~
.. autoclass:: SplitChannel
    :members:
    :special-members: __call__

`Activations`
~~~~~~~~~~~~~
.. autoclass:: Activations
    :members:
    :special-members: __call__

`AsDiscrete`
~~~~~~~~~~~~
.. autoclass:: AsDiscrete
    :members:
    :special-members: __call__

`KeepLargestConnectedComponent`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: KeepLargestConnectedComponent
    :members:
    :special-members: __call__


Dictionary-based Transforms
---------------------------

`Spacingd`
~~~~~~~~~~
.. autoclass:: Spacingd
    :members:
    :special-members: __call__

`Orientationd`
~~~~~~~~~~~~~~
.. autoclass:: Orientationd
    :members:
    :special-members: __call__

`LoadNiftid`
~~~~~~~~~~~~
.. autoclass:: LoadNiftid
    :members:
    :special-members: __call__

`LoadPNGd`
~~~~~~~~~~
.. autoclass:: LoadPNGd
    :members:
    :special-members: __call__

`Identityd`
~~~~~~~~~~~
.. autoclass:: Identityd
    :members:
    :special-members: __call__

`AsChannelFirstd`
~~~~~~~~~~~~~~~~~
.. autoclass:: AsChannelFirstd
    :members:
    :special-members: __call__

`AsChannelLastd`
~~~~~~~~~~~~~~~~
.. autoclass:: AsChannelLastd
    :members:
    :special-members: __call__

`AddChanneld`
~~~~~~~~~~~~~
.. autoclass:: AddChanneld
    :members:
    :special-members: __call__

`RepeatChanneld`
~~~~~~~~~~~~~~~~
.. autoclass:: RepeatChanneld
    :members:
    :special-members: __call__

`CastToTyped`
~~~~~~~~~~~~~
.. autoclass:: CastToTyped
    :members:
    :special-members: __call__

`ToTensord`
~~~~~~~~~~~
.. autoclass:: ToTensord
    :members:
    :special-members: __call__

`ToNumpyd`
~~~~~~~~~~
.. autoclass:: ToNumpyd
    :members:
    :special-members: __call__

`Rotate90d`
~~~~~~~~~~~
.. autoclass:: Rotate90d
    :members:
    :special-members: __call__

`Resized`
~~~~~~~~~
.. autoclass:: Resized
    :members:
    :special-members: __call__

`RandGaussianNoised`
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: RandGaussianNoised
    :members:
    :special-members: __call__

`RandRotate90d`
~~~~~~~~~~~~~~~
.. autoclass:: RandRotate90d
    :members:
    :special-members: __call__

`ShiftIntensityd`
~~~~~~~~~~~~~~~~~
.. autoclass:: ShiftIntensityd
    :members:
    :special-members: __call__

`RandShiftIntensityd`
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: RandShiftIntensityd
    :members:
    :special-members: __call__

`ScaleIntensityd`
~~~~~~~~~~~~~~~~~
.. autoclass:: ScaleIntensityd
    :members:
    :special-members: __call__

`RandScaleIntensityd`
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: RandScaleIntensityd
    :members:
    :special-members: __call__

`NormalizeIntensityd`
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: NormalizeIntensityd
    :members:
    :special-members: __call__

`ThresholdIntensityd`
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ThresholdIntensityd
    :members:
    :special-members: __call__

`ScaleIntensityRanged`
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ScaleIntensityRanged
    :members:
    :special-members: __call__

`AdjustContrastd`
~~~~~~~~~~~~~~~~~
.. autoclass:: AdjustContrastd
    :members:
    :special-members: __call__

`RandAdjustContrastd`
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: RandAdjustContrastd
    :members:
    :special-members: __call__

`SpatialPadd`
~~~~~~~~~~~~~
.. autoclass:: SpatialPadd
    :members:
    :special-members: __call__

`BorderPadd`
~~~~~~~~~~~~
.. autoclass:: BorderPadd
    :members:
    :special-members: __call__

`DivisiblePadd`
~~~~~~~~~~~~~~~
.. autoclass:: DivisiblePadd
    :members:
    :special-members: __call__

`SpatialCropd`
~~~~~~~~~~~~~~
.. autoclass:: SpatialCropd
    :members:
    :special-members: __call__

`CenterSpatialCropd`
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: CenterSpatialCropd
    :members:
    :special-members: __call__

`RandSpatialCropd`
~~~~~~~~~~~~~~~~~~
.. autoclass:: RandSpatialCropd
    :members:
    :special-members: __call__

`RandSpatialCropSamplesd`
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: RandSpatialCropSamplesd
    :members:
    :special-members: __call__

`CropForegroundd`
~~~~~~~~~~~~~~~~~
.. autoclass:: CropForegroundd
    :members:
    :special-members: __call__

`RandCropByPosNegLabeld`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: RandCropByPosNegLabeld
    :members:
    :special-members: __call__

`RandAffined`
~~~~~~~~~~~~~
.. autoclass:: RandAffined
    :members:
    :special-members: __call__

`Rand2DElasticd`
~~~~~~~~~~~~~~~~
.. autoclass:: Rand2DElasticd
    :members:
    :special-members: __call__

`Rand3DElasticd`
~~~~~~~~~~~~~~~~
.. autoclass:: Rand3DElasticd
    :members:
    :special-members: __call__

`Flipd`
~~~~~~~
.. autoclass:: Flipd
    :members:
    :special-members: __call__

`RandFlipd`
~~~~~~~~~~~
.. autoclass:: RandFlipd
    :members:
    :special-members: __call__

`Rotated`
~~~~~~~~~
.. autoclass:: Rotated
    :members:
    :special-members: __call__

`RandRotated`
~~~~~~~~~~~~~
.. autoclass:: RandRotated
    :members:
    :special-members: __call__

`Zoomd`
~~~~~~~
.. autoclass:: Zoomd
    :members:
    :special-members: __call__

`RandZoomd`
~~~~~~~~~~~
.. autoclass:: RandZoomd
    :members:
    :special-members: __call__

`DeleteItemsd`
~~~~~~~~~~~~~~
.. autoclass:: DeleteItemsd
    :members:
    :special-members: __call__

`SqueezeDimd`
~~~~~~~~~~~~~
.. autoclass:: SqueezeDimd
    :members:
    :special-members: __call__

`DataStatsd`
~~~~~~~~~~~~
.. autoclass:: DataStatsd
    :members:
    :special-members: __call__

`SimulateDelayd`
~~~~~~~~~~~~~~~~
.. autoclass:: SimulateDelayd
    :members:
    :special-members: __call__

`SplitChanneld`
~~~~~~~~~~~~~~~
.. autoclass:: SplitChanneld
    :members:
    :special-members: __call__

`Activationsd`
~~~~~~~~~~~~~~
.. autoclass:: Activationsd
    :members:
    :special-members: __call__

`AsDiscreted`
~~~~~~~~~~~~~
.. autoclass:: AsDiscreted
    :members:
    :special-members: __call__

`KeepLargestConnectedComponentd`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: KeepLargestConnectedComponentd
    :members:
    :special-members: __call__

`CopyItemsd`
~~~~~~~~~~~~
.. autoclass:: CopyItemsd
    :members:
    :special-members: __call__

`ConcatItemsd`
~~~~~~~~~~~~~~
.. autoclass:: ConcatItemsd
    :members:
    :special-members: __call__


Transform Adaptors
------------------
.. automodule:: monai.transforms.adaptors

`adaptor`
~~~~~~~~~
.. autofunction:: monai.transforms.adaptors.adaptor

`apply_alias`
~~~~~~~~~~~~~
.. autofunction:: monai.transforms.adaptors.apply_alias

`to_kwargs`
~~~~~~~~~~~
.. autofunction:: monai.transforms.adaptors.to_kwargs


Utilities
---------
.. automodule:: monai.transforms.utils
    :members:
