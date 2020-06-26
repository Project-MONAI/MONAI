:github_url: https://github.com/Project-MONAI/MONAI

.. _networks:

Network architectures
=====================

Blocks
------
.. automodule:: monai.networks.blocks
.. currentmodule:: monai.networks.blocks

`Convolution`
~~~~~~~~~~~~~
.. autoclass:: Convolution
    :members:

`ResidualUnit`
~~~~~~~~~~~~~~
.. autoclass:: ResidualUnit
    :members:

`Squeeze-and-Excitation`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ChannelSELayer
    :members:

`Residual Squeeze-and-Excitation`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ResidualSELayer
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


Layers
------

`Factories`
~~~~~~~~~~~
.. automodule:: monai.networks.layers.factories
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

`Utilities`
~~~~~~~~~~~
.. automodule:: monai.networks.layers.convutils
    :members:


Nets
----
.. currentmodule:: monai.networks.nets

`Densenet3D`
~~~~~~~~~~~~
.. autoclass:: DenseNet
  :members:
.. autofunction:: densenet121
.. autofunction:: densenet169
.. autofunction:: densenet201
.. autofunction:: densenet264

`Highresnet`
~~~~~~~~~~~~
.. autoclass:: HighResNet
  :members:
.. autoclass:: HighResBlock
  :members:

`Unet`
~~~~~~
.. autoclass:: UNet
  :members:
.. autoclass:: Unet
.. autoclass:: unet

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
.. autoclass:: critic
  :members:

Utilities
---------
.. automodule:: monai.networks.utils
  :members:
