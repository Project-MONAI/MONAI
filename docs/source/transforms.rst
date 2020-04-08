:github_url: https://github.com/Project-MONAI/MONAI

.. _transform_api:

Transforms
==========


Generic Interfaces
------------------

.. automodule:: monai.transforms.compose
.. currentmodule:: monai.transforms.compose


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

.. automodule:: monai.transforms.transforms
.. currentmodule:: monai.transforms.transforms

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

`Transpose`
~~~~~~~~~~~
.. autoclass:: Transpose
    :members:
    :special-members: __call__

`Rescale`
~~~~~~~~~
.. autoclass:: Rescale
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

`ToTensor`
~~~~~~~~~~
.. autoclass:: ToTensor
    :members:
    :special-members: __call__

`RandUniformPatch`
~~~~~~~~~~~~~~~~~~
.. autoclass:: RandUniformPatch
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

`PadImageEnd`
~~~~~~~~~~~~~
.. autoclass:: PadImageEnd
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

`SpatialCrop`
~~~~~~~~~~~~~
.. autoclass:: SpatialCrop
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


Dictionary-based Composables
----------------------------

.. automodule:: monai.transforms.composables
.. currentmodule:: monai.transforms.composables

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

`Rotate90d`
~~~~~~~~~~~
.. autoclass:: Rotate90d
    :members:
    :special-members: __call__

`SpatialCropd`
~~~~~~~~~~~~~~
.. autoclass:: SpatialCropd
    :members:
    :special-members: __call__

`CropForegroundd`
~~~~~~~~~~~~~~~~~
.. autoclass:: CropForegroundd
    :members:
    :special-members: __call__

`Rescaled`
~~~~~~~~~~
.. autoclass:: Rescaled
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

`RandUniformPatchd`
~~~~~~~~~~~~~~~~~~~
.. autoclass:: RandUniformPatchd
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

`RandRotate90d`
~~~~~~~~~~~~~~~
.. autoclass:: RandRotate90d
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

`DeleteKeysd`
~~~~~~~~~~~~~
.. autoclass:: DeleteKeysd
    :members:
    :special-members: __call__

Transform Adaptors
------------------

.. automodule:: monai.transforms.adaptors
.. currentmodule:: monai.transforms.adaptors

`adaptor`
~~~~~~~~~
.. automethod:: monai.transforms.adaptors.adaptor

`apply_alias`
~~~~~~~~~~~~~
.. automethod:: monai.transforms.adaptors.apply_alias


`to_kwargs`
~~~~~~~~~~~
.. automethod:: monai.transforms.adaptors.to_kwargs
