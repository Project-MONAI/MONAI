:github_url: https://github.com/Project-MONAI/MONAI

.. _networkss:

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


Nets
----

.. automodule:: monai.networks.nets
.. currentmodule:: monai.networks.nets


`Densenet3D`
~~~~~~~~~~~~
.. automodule:: monai.networks.nets.densenet3d
  :members:
.. automethod:: monai.networks.nets.densenet3d.densenet121
.. automethod:: monai.networks.nets.densenet3d.densenet169
.. automethod:: monai.networks.nets.densenet3d.densenet201
.. automethod:: monai.networks.nets.densenet3d.densenet264

`Highresnet`
~~~~~~~~~~~~
.. automodule:: monai.networks.nets.highresnet
  :members:

`Unet`
~~~~~~
.. automodule:: monai.networks.nets.unet
  :members:
