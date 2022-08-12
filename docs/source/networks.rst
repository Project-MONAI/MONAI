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

`CRF`
~~~~~
.. autoclass:: CRF
    :members:

`ResidualUnit`
~~~~~~~~~~~~~~
.. autoclass:: ResidualUnit
    :members:

`Swish`
~~~~~~~
.. autoclass:: Swish
    :members:

`MemoryEfficientSwish`
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: MemoryEfficientSwish
    :members:

`FPN`
~~~~~
.. autoclass:: ExtraFPNBlock
    :members:
.. autoclass:: FeaturePyramidNetwork
    :members:
.. autoclass:: LastLevelMaxPool
    :members:
.. autoclass:: LastLevelP6P7
    :members:
.. autoclass:: BackboneWithFPN
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
.. autoclass:: UnetOutBlock
    :members:

`SegResnet Block`
~~~~~~~~~~~~~~~~~
.. autoclass:: ResBlock
    :members:

`SABlock Block`
~~~~~~~~~~~~~~~
.. autoclass:: SABlock
    :members:

`Squeeze-and-Excitation`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ChannelSELayer
    :members:

`Transformer Block`
~~~~~~~~~~~~~~~~~~~
.. autoclass:: TransformerBlock
    :members:

`UNETR Block`
~~~~~~~~~~~~~
.. autoclass:: UnetrBasicBlock
    :members:
.. autoclass:: UnetrUpBlock
    :members:
.. autoclass:: UnetrPrUpBlock
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

`Squeeze-and-Excitation ResNeXt Bottleneck`
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

`Registration Residual Conv Block`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: RegistrationResidualConvBlock
    :members:

`Registration Down Sample Block`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: RegistrationDownSampleBlock
    :members:

`Registration Extraction Block`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: RegistrationExtractionBlock
    :members:

`LocalNet DownSample Block`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: LocalNetDownSampleBlock
    :members:

`LocalNet UpSample Block`
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: LocalNetUpSampleBlock
    :members:

`LocalNet Feature Extractor Block`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: LocalNetFeatureExtractorBlock
    :members:

`MLP Block`
~~~~~~~~~~~
.. autoclass:: MLPBlock
    :members:

`Patch Embedding Block`
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: PatchEmbeddingBlock
    :members:

`FactorizedIncreaseBlock`
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: FactorizedIncreaseBlock
    :members:

`FactorizedReduceBlock`
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: FactorizedReduceBlock
    :members:

`P3DActiConvNormBlock`
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: P3DActiConvNormBlock
    :members:

`ActiConvNormBlock`
~~~~~~~~~~~~~~~~~~~
.. autoclass:: ActiConvNormBlock
    :members:

`Warp`
~~~~~~
.. autoclass:: Warp
    :members:

`DVF2DDF`
~~~~~~~~~
.. autoclass:: DVF2DDF
    :members:

`VarNetBlock`
~~~~~~~~~~~~~
.. autoclass:: monai.apps.reconstruction.networks.blocks.varnetblock.VarNetBlock
   :members:


N-Dim Fourier Transform
~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: monai.networks.blocks.fft_utils_t
.. autofunction:: monai.networks.blocks.fft_utils_t.fftn_centered_t
.. autofunction:: monai.networks.blocks.fft_utils_t.ifftn_centered_t
.. autofunction:: monai.networks.blocks.fft_utils_t.roll
.. autofunction:: monai.networks.blocks.fft_utils_t.roll_1d
.. autofunction:: monai.networks.blocks.fft_utils_t.fftshift
.. autofunction:: monai.networks.blocks.fft_utils_t.ifftshift

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

`Pad`
~~~~~
.. automodule:: monai.networks.layers.Pad
  :members:

`Pool`
~~~~~~
.. automodule:: monai.networks.layers.Pool
  :members:

.. currentmodule:: monai.networks.layers

`ChannelPad`
~~~~~~~~~~~~
.. autoclass:: ChannelPad
    :members:

`SkipConnection`
~~~~~~~~~~~~~~~~
.. autoclass:: SkipConnection
    :members:

`Flatten`
~~~~~~~~~
.. autoclass:: Flatten
    :members:

`Reshape`
~~~~~~~~~
.. autoclass:: Reshape
    :members:

`separable_filtering`
~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: separable_filtering

`apply_filter`
~~~~~~~~~~~~~~
.. autofunction:: apply_filter

`GaussianFilter`
~~~~~~~~~~~~~~~~
.. autoclass:: GaussianFilter
    :members:

`BilateralFilter`
~~~~~~~~~~~~~~~~~
.. autoclass:: BilateralFilter
    :members:

`PHLFilter`
~~~~~~~~~~~
.. autoclass:: PHLFilter

`GaussianMixtureModel`
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: GaussianMixtureModel

`SavitzkyGolayFilter`
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: SavitzkyGolayFilter
    :members:

`HilbertTransform`
~~~~~~~~~~~~~~~~~~
.. autoclass:: HilbertTransform
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
.. automodule:: monai.networks.layers.utils
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

`DenseNet121`
~~~~~~~~~~~~~
.. autoclass:: DenseNet121

`DenseNet169`
~~~~~~~~~~~~~
.. autoclass:: DenseNet169

`DenseNet201`
~~~~~~~~~~~~~
.. autoclass:: DenseNet201

`DenseNet264`
~~~~~~~~~~~~~
.. autoclass:: DenseNet264

`EfficientNet`
~~~~~~~~~~~~~~
.. autoclass:: EfficientNet
  :members:

`BlockArgs`
~~~~~~~~~~~
.. autoclass:: BlockArgs
  :members:

`EfficientNetBN`
~~~~~~~~~~~~~~~~
.. autoclass:: EfficientNetBN
  :members:

`EfficientNetBNFeatures`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: EfficientNetBNFeatures
  :members:

`SegResNet`
~~~~~~~~~~~
.. autoclass:: SegResNet
  :members:

`SegResNetVAE`
~~~~~~~~~~~~~~
.. autoclass:: SegResNetVAE
  :members:

`ResNet`
~~~~~~~~
.. autoclass:: ResNet
  :members:

`SENet`
~~~~~~~
.. autoclass:: SENet
  :members:

`SENet154`
~~~~~~~~~~
.. autoclass:: SENet154

`SEResNet50`
~~~~~~~~~~~~
.. autoclass:: SEResNet50

`SEResNet101`
~~~~~~~~~~~~~
.. autoclass:: SEResNet101

`SEResNet152`
~~~~~~~~~~~~~
.. autoclass:: SEResNet152

`SEResNext50`
~~~~~~~~~~~~~
.. autoclass:: SEResNext50

`SEResNext101`
~~~~~~~~~~~~~~
.. autoclass:: SEResNext101

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

`AttentionUnet`
~~~~~~~~~~~~~~~
.. autoclass:: AttentionUnet
  :members:

`UNETR`
~~~~~~~
.. autoclass:: UNETR
    :members:

`SwinUNETR`
~~~~~~~~~~~
.. autoclass:: SwinUNETR
    :members:

`BasicUNet`
~~~~~~~~~~~
.. autoclass:: BasicUNet
  :members:
.. autoclass:: BasicUnet
.. autoclass:: Basicunet

`FlexibleUNet`
~~~~~~~~~~~~~~
.. autoclass:: FlexibleUNet
  :members:

`VNet`
~~~~~~
.. autoclass:: VNet
  :members:

`RegUNet`
~~~~~~~~~
.. autoclass:: RegUNet
  :members:

`GlobalNet`
~~~~~~~~~~~~
.. autoclass:: GlobalNet
  :members:

`LocalNet`
~~~~~~~~~~~
.. autoclass:: LocalNet
  :members:

`AutoEncoder`
~~~~~~~~~~~~~
.. autoclass:: AutoEncoder
  :members:

`VarAutoEncoder`
~~~~~~~~~~~~~~~~
.. autoclass:: VarAutoEncoder
  :members:

`ViT`
~~~~~
.. autoclass:: ViT
  :members:

`ViTAutoEnc`
~~~~~~~~~~~~
.. autoclass:: ViTAutoEnc
  :members:

`FullyConnectedNet`
~~~~~~~~~~~~~~~~~~~
.. autoclass:: FullyConnectedNet
  :members:

`VarFullyConnectedNet`
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: VarFullyConnectedNet
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

`Transchex`
~~~~~~~~~~~~~~~~
.. autoclass:: Transchex
  :members:

`NetAdapter`
~~~~~~~~~~~~
.. autoclass:: NetAdapter
  :members:

`TorchVisionFCModel`
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: TorchVisionFCModel
  :members:

`MILModel`
~~~~~~~~~~
.. autoclass:: MILModel
  :members:

`DiNTS`
~~~~~~~
.. autoclass:: DiNTS
  :members:

`TopologyConstruction for DiNTS`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: TopologyConstruction
  :members:

`TopologyInstance for DiNTS`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: TopologyInstance
  :members:

`TopologySearch for DiNTS`
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: TopologySearch
  :members:

`ComplexUnet`
~~~~~~~~~~~~~
.. autoclass:: monai.apps.reconstruction.networks.nets.complex_unet.ComplexUnet
   :members:

`CoilSensitivityModel`
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: monai.apps.reconstruction.networks.nets.coil_sensitivity_model.CoilSensitivityModel
   :members:

`e2e-VarNet`
~~~~~~~~~~~~
.. autoclass:: monai.apps.reconstruction.networks.nets.varnet.VariationalNetworkModel
   :members:

Utilities
---------
.. automodule:: monai.networks.utils
  :members:

.. automodule:: monai.apps.reconstruction.networks.nets.utils
  :members:
