:github_url: https://github.com/Project-MONAI/MONAI

.. _networks:

Network architectures
=====================


Blocks
------
.. automodule:: monai.networks.blocks.convolutions
.. currentmodule:: monai.networks.blocks.convolutions


`Convolution`
~~~~~~~~~~~~~
.. autoclass:: Convolution
    :members:

`ResidualUnit`
~~~~~~~~~~~~~~
.. autoclass:: ResidualUnit
    :members:


Layers
------

`Factories`
~~~~~~~~~~~
.. automodule:: monai.networks.layers.factories
.. currentmodule:: monai.networks.layers.factories

`LayerFactory`
##############
.. autoclass:: LayerFactory

.. automodule:: monai.networks.layers.simplelayers
.. currentmodule:: monai.networks.layers.simplelayers

`SkipConnection`
~~~~~~~~~~~~~~~~
.. autoclass:: SkipConnection
    :members:

`Flatten`
~~~~~~~~~~
.. autoclass:: Flatten
    :members:

`GaussianFilter`
~~~~~~~~~~~~~~~~
.. autoclass:: GaussianFilter
    :members:
    :special-members: __call__

`Utilities`
~~~~~~~~~~~
.. automodule:: monai.networks.layers.convutils
    :members:


Nets
----

.. automodule:: monai.networks.nets
.. currentmodule:: monai.networks.nets


`Densenet3D`
~~~~~~~~~~~~
.. automodule:: monai.networks.nets.densenet
  :members:
.. automethod:: monai.networks.nets.densenet.densenet121
.. automethod:: monai.networks.nets.densenet.densenet169
.. automethod:: monai.networks.nets.densenet.densenet201
.. automethod:: monai.networks.nets.densenet.densenet264

`Highresnet`
~~~~~~~~~~~~
.. automodule:: monai.networks.nets.highresnet
  :members:

`Unet`
~~~~~~
.. automodule:: monai.networks.nets.unet
  :members:


Utilities
---------
.. automodule:: monai.networks.utils
  :members:
