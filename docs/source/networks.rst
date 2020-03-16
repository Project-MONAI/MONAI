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

`get_conv_type`
~~~~~~~~~~~~~~~
.. automethod:: monai.networks.layers.factories.get_conv_type

`get_dropout_type`
~~~~~~~~~~~~~~~~~~
.. automethod:: monai.networks.layers.factories.get_dropout_type

`get_normalize_type`
~~~~~~~~~~~~~~~~~~~~
.. automethod:: monai.networks.layers.factories.get_normalize_type

`get_maxpooling_type`
~~~~~~~~~~~~~~~~~~~~~
.. automethod:: monai.networks.layers.factories.get_maxpooling_type

`get_avgpooling_type`
~~~~~~~~~~~~~~~~~~~~~
.. automethod:: monai.networks.layers.factories.get_avgpooling_type


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
