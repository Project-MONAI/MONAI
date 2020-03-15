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

`MapTransform`
~~~~~~~~~~~~~~
.. autoclass:: MapTransform
    :members:

`Randomizable`
~~~~~~~~~~~~~~
.. autoclass:: Randomizable
    :members:

`Compose`
~~~~~~~~~
.. autoclass:: Compose
    :members:


Vanilla Transforms
------------------

.. automodule:: monai.transforms.transforms
.. currentmodule:: monai.transforms.transforms

`Spacing`
~~~~~~~~~
.. autoclass:: Spacing
    :members:

`Orientation`
~~~~~~~~~~~~~
.. autoclass:: Orientation
    :members:

`LoadNifti`
~~~~~~~~~~~
.. autoclass:: LoadNifti
    :members:

`AsChannelFirst`
~~~~~~~~~~~~~~~~
.. autoclass:: AsChannelFirst
    :members:

`AddChannel`
~~~~~~~~~~~~
.. autoclass:: AddChannel
    :members:

`Transpose`
~~~~~~~~~~~
.. autoclass:: Transpose
    :members:

`Rescale`
~~~~~~~~~
.. autoclass:: Rescale
    :members:

`GaussianNoise`
~~~~~~~~~~~~~~~
.. autoclass:: GaussianNoise
    :members:

`Flip`
~~~~~~
.. autoclass:: Flip
    :members:

`Resize`
~~~~~~~~
.. autoclass:: Resize
    :members:

`Rotate`
~~~~~~~~
.. autoclass:: Rotate
    :members:

`Zoom`
~~~~~~
.. autoclass:: Zoom
    :members:

`ToTensor`
~~~~~~~~~~
.. autoclass:: ToTensor
    :members:

`UniformRandomPatch`
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: UniformRandomPatch
    :members:

`IntensityNormalizer`
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: IntensityNormalizer
    :members:

`ImageEndPadder`
~~~~~~~~~~~~~~~~
.. autoclass:: ImageEndPadder
    :members:

`Rotate90`
~~~~~~~~~~
.. autoclass:: Rotate90
    :members:

`RandRotate90`
~~~~~~~~~~~~~~
.. autoclass:: RandRotate90
    :members:

`SpatialCrop`
~~~~~~~~~~~~~
.. autoclass:: SpatialCrop
    :members:

`RandRotate`
~~~~~~~~~~~~
.. autoclass:: RandRotate
    :members:

`RandFlip`
~~~~~~~~~~
.. autoclass:: RandFlip
    :members:

`RandZoom`
~~~~~~~~~~
.. autoclass:: RandZoom
    :members:

`Affine`
~~~~~~~~
.. autoclass:: Affine
    :members:

`RandAffine`
~~~~~~~~~~~~
.. autoclass:: RandAffine
    :members:

`Rand2DElastic`
~~~~~~~~~~~~~~~
.. autoclass:: Rand2DElastic
    :members:

`Rand3DElastic`
~~~~~~~~~~~~~~~
.. autoclass:: Rand3DElastic
    :members:


Dictionary-based Composables
----------------------------

.. automodule:: monai.transforms.composables
.. currentmodule:: monai.transforms.composables

`Spacingd`
~~~~~~~~~~
.. autoclass:: Spacingd
    :members:

`Orientationd`
~~~~~~~~~~~~~~
.. autoclass:: Orientationd
    :members:

`LoadNiftid`
~~~~~~~~~~~~
.. autoclass:: LoadNiftid
    :members:

`AsChannelFirstd`
~~~~~~~~~~~~~~~~~
.. autoclass:: AsChannelFirstd
    :members:

`AddChanneld`
~~~~~~~~~~~~~
.. autoclass:: AddChanneld
    :members:

`Rotate90d`
~~~~~~~~~~~
.. autoclass:: Rotate90d
    :members:

`Rescaled`
~~~~~~~~~~
.. autoclass:: Rescaled
    :members:

`Resized`
~~~~~~~~~
.. autoclass:: Resized
    :members:

`UniformRandomPatchd`
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: UniformRandomPatchd
    :members:

`RandRotate90d`
~~~~~~~~~~~~~~~
.. autoclass:: RandRotate90d
    :members:

`RandCropByPosNegLabeld`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: RandCropByPosNegLabeld
    :members:

`RandAffined`
~~~~~~~~~~~~~
.. autoclass:: RandAffined
    :members:

`Rand2DElasticd`
~~~~~~~~~~~~~~~~
.. autoclass:: Rand2DElasticd
    :members:

`Rand3DElasticd`
~~~~~~~~~~~~~~~~
.. autoclass:: Rand3DElasticd
    :members:

`Flipd`
~~~~~~~
.. autoclass:: Flipd
    :members:

`RandFlipd`
~~~~~~~~~~~
.. autoclass:: RandFlipd
    :members:

`Rotated`
~~~~~~~~~
.. autoclass:: Rotated
    :members:

`RandRotated`
~~~~~~~~~~~~~
.. autoclass:: RandRotated
    :members:

`Zoomd`
~~~~~~~
.. autoclass:: Zoomd
    :members:

`RandZoomd`
~~~~~~~~~~~
.. autoclass:: RandZoomd
    :members:


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