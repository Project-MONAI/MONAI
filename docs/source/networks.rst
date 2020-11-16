:github_url: https://github.com/Project-MONAI/MONAI

.. _networks:

Network architectures
=====================

Blocks
------
.. automodule:: monai.networks.blocks
.. currentmodule:: monai.networks.blocks

`ADN`
~~~~~
.. autoclass:: ADN
    :members:

`Convolution`
~~~~~~~~~~~~~
.. autoclass:: Convolution
    :members:

`ResidualUnit`
~~~~~~~~~~~~~~
.. autoclass:: ResidualUnit
    :members:

`Swish`
~~~~~~~
.. autoclass:: Swish
    :members:

`Mish`
~~~~~~
.. autoclass:: Mish
    :members:

`GCN Module`
~~~~~~~~~~~~
.. autoclass:: GCN
    :members:

`Refinement Module`
~~~~~~~~~~~~~~~~~~~
.. autoclass:: Refine
    :members:

`FCN Module`
~~~~~~~~~~~~
.. autoclass:: FCN
    :members:

`Multi-Channel FCN Module`
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: MCFCN
    :members:

`Dynamic-Unet Block`
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: UnetResBlock
    :members:
.. autoclass:: UnetBasicBlock
    :members:
.. autoclass:: UnetUpBlock
    :members:

`SegResnet Block`
~~~~~~~~~~~~~~~~~
.. autoclass:: ResBlock
    :members:

`Squeeze-and-Excitation`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ChannelSELayer
    :members:

`Residual Squeeze-and-Excitation`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ResidualSELayer
    :members:

`Squeeze-and-Excitation Block`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: SEBlock
    :members:

`Squeeze-and-Excitation Bottleneck`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: SEBottleneck
    :members:

`Squeeze-and-Excitation Resnet Bottleneck`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: SEResNetBottleneck
    :members:

`Squeeze-and-Excitation ResneXt Bottleneck`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: SEResNeXtBottleneck
    :members:

`Simple ASPP`
~~~~~~~~~~~~~
.. autoclass:: SimpleASPP
    :members:

`MaxAvgPooling`
~~~~~~~~~~~~~~~
.. autoclass:: MaxAvgPool
    :members:

`Upsampling`
~~~~~~~~~~~~
.. autoclass:: UpSample
    :members:
.. autoclass:: Upsample
.. autoclass:: SubpixelUpsample
    :members:
.. autoclass:: Subpixelupsample
.. autoclass:: SubpixelUpSample



Layers
------

`Factories`
~~~~~~~~~~~
.. automodule:: monai.networks.layers.factories

.. autoclass:: monai.networks.layers.LayerFactory
  :members:

.. currentmodule:: monai.networks.layers

`split_args`
~~~~~~~~~~~~
.. autofunction:: monai.networks.layers.split_args

`Dropout`
~~~~~~~~~
.. automodule:: monai.networks.layers.Dropout
  :members:

`Act`
~~~~~
.. automodule:: monai.networks.layers.Act
  :members:

`Norm`
~~~~~~
.. automodule:: monai.networks.layers.Norm
  :members:

`Conv`
~~~~~~
.. automodule:: monai.networks.layers.Conv
  :members:

`Pool`
~~~~~~
.. automodule:: monai.networks.layers.Pool
  :members:

.. currentmodule:: monai.networks.layers

`SkipConnection`
~~~~~~~~~~~~~~~~
.. autoclass:: SkipConnection
    :members:

`Flatten`
~~~~~~~~~
.. autoclass:: Flatten
    :members:

`GaussianFilter`
~~~~~~~~~~~~~~~~
.. autoclass:: GaussianFilter
    :members:

`Affine Transform`
~~~~~~~~~~~~~~~~~~
.. autoclass:: monai.networks.layers.AffineTransform
    :members:

`grid_pull`
~~~~~~~~~~~
.. autofunction:: monai.networks.layers.grid_pull

`grid_push`
~~~~~~~~~~~
.. autofunction:: monai.networks.layers.grid_push

`grid_count`
~~~~~~~~~~~~
.. autofunction:: monai.networks.layers.grid_count

`grid_grad`
~~~~~~~~~~~
.. autofunction:: monai.networks.layers.grid_grad

`LLTM`
~~~~~~
.. autoclass:: LLTM
    :members:

`Utilities`
~~~~~~~~~~~
.. automodule:: monai.networks.layers.convutils
    :members:


Nets
----
.. currentmodule:: monai.networks.nets

`AHNet`
~~~~~~~
.. autoclass:: AHNet
  :members:

`DenseNet`
~~~~~~~~~~
.. autoclass:: DenseNet
  :members:
.. autofunction:: densenet121
.. autofunction:: densenet169
.. autofunction:: densenet201
.. autofunction:: densenet264

`SegResNet`
~~~~~~~~~~~
.. autoclass:: SegResNet
  :members:

`SegResNetVAE`
~~~~~~~~~~~~~~
.. autoclass:: SegResNetVAE
  :members:

`SENet`
~~~~~~~
.. autoclass:: SENet
  :members:
.. autofunction:: senet154
.. autofunction:: se_resnet50
.. autofunction:: se_resnet101
.. autofunction:: se_resnet152
.. autofunction:: se_resnext50_32x4d
.. autofunction:: se_resnext101_32x4d

`HighResNet`
~~~~~~~~~~~~
.. autoclass:: HighResNet
  :members:
.. autoclass:: HighResBlock
  :members:

`DynUNet`
~~~~~~~~~
.. autoclass:: DynUNet
  :members:
.. autoclass:: DynUnet
.. autoclass:: Dynunet

`UNet`
~~~~~~
.. autoclass:: UNet
  :members:
.. autoclass:: Unet
.. autoclass:: unet

`BasicUNet`
~~~~~~~~~~~
.. autoclass:: BasicUNet
  :members:
.. autoclass:: BasicUnet
.. autoclass:: Basicunet

`VNet`
~~~~~~
.. autoclass:: VNet
  :members:

`AutoEncoder`
~~~~~~~~~~~~~
.. autoclass:: AutoEncoder
  :members:

`VarAutoEncoder`
~~~~~~~~~~~~~~~~
.. autoclass:: VarAutoEncoder
  :members:

`FullyConnectedNet`
~~~~~~~~~~~~~~~~~~~
.. autoclass:: FullyConnectedNet
  :members:

`Generator`
~~~~~~~~~~~
.. autoclass:: Generator
  :members:

`Regressor`
~~~~~~~~~~~
.. autoclass:: Regressor
  :members:

`Classifier`
~~~~~~~~~~~~
.. autoclass:: Classifier
  :members:

`Discriminator`
~~~~~~~~~~~~~~~
.. autoclass:: Discriminator
  :members:

`Critic`
~~~~~~~~
.. autoclass:: Critic
  :members:

Utilities
---------
.. automodule:: monai.networks.utils
  :members:
