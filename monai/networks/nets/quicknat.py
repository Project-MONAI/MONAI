# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import ConvDenseBlock, Convolution
from monai.networks.blocks import squeeze_and_excitation as se
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.networks.layers.utils import get_dropout_layer, get_pool_layer
from monai.utils import optional_import

# Lazy import to avoid dependency
se1, flag = optional_import("squeeze_and_excitation")

__all__ = ["Quicknat"]

# QuickNAT specific Blocks


class SkipConnectionWithIdx(SkipConnection):
    """
    Combine the forward pass input with the result from the given submodule::
    --+--submodule--o--
      |_____________|
    The available modes are ``"cat"``, ``"add"``, ``"mul"``.
    Defaults to "cat" and dimension 1.
    Inherits from SkipConnection but provides the indizes with each forward pass.
    """

    def forward(self, input, indices):  # type: ignore[override]
        return super().forward(input), indices


class SequentialWithIdx(nn.Sequential):
    """
    A sequential container.
    Modules will be added to it in the order they are passed in the
    constructor.
    Own implementation to work with the new indices in the forward pass.
    """

    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, input, indices):  # type: ignore[override]
        for module in self:
            input, indices = module(input, indices)
        return input, indices


class ClassifierBlock(Convolution):
    """
    Returns a classifier block without an activation function at the top.
    It consists of a 1 * 1 convolutional layer which maps the input to a num_class channel feature map.
    The output is a probability map for each of the classes.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of classes to map to.
        strides: convolution stride. Defaults to 1.
        kernel_size: convolution kernel size. Defaults to 3.
        adn_ordering: a string representing the ordering of activation, normalization, and dropout.
        Defaults to "NDA".
        act: activation type and arguments. Defaults to PReLU.

    """

    def __init__(self, spatial_dims, in_channels, out_channels, strides, kernel_size, act=None, adn_ordering="A"):
        super().__init__(spatial_dims, in_channels, out_channels, strides, kernel_size, adn_ordering, act)

    def forward(self, input: torch.Tensor, weights=None, indices=None):
        _, channel, *dims = input.size()
        if weights is not None:
            weights, _ = torch.max(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            # use weights to adapt how the classes are weighted.
            if len(dims) == 2:
                out_conv = F.conv2d(input, weights)
            else:
                raise ValueError("Quicknat is a 2D architecture, please check your dimension.")
        else:
            out_conv = super().forward(input)
        # no indices to return
        return out_conv, None


# Quicknat specific blocks. All blocks inherit from MONAI blocks but have adaptions to their structure
class ConvConcatDenseBlock(ConvDenseBlock):
    """
    This dense block is defined as a sequence of 'Convolution' blocks. It overwrite the '_get_layer' methodto change the ordering of
    Every convolutional layer is preceded by a batch-normalization layer and a Rectifier Linear Unit (ReLU) layer.
    The first two convolutional layers are followed by a concatenation layer that concatenates
    the input feature map with outputs of the current and previous convolutional blocks.
    Kernel size of two convolutional layers kept small to limit number of paramters.
    Appropriate padding is provided so that the size of feature maps before and after convolution remains constant.
    The output channels for each convolution layer is set to 64, which acts as a bottle- neck for feature map selectivity.
    The input channel size is variable, depending on the number of dense connections.
    The third convolutional layer is also preceded by a batch normalization and ReLU,
    but has a 1 * 1 kernel size to compress the feature map size to 64.
    Args:
        in_channles: variable depending on depth of the network
        seLayer: Squeeze and Excite block to be included, defaults to None, valid options are {'NONE', 'CSE', 'SSE', 'CSSE'},
        dropout_layer: Dropout block to be included, defaults to None.
    :return: forward passed tensor
    """

    def __init__(
        self,
        in_channels: int,
        se_layer: Optional[nn.Module] = None,
        dropout_layer: Optional[nn.Dropout2d] = None,
        kernel_size: Sequence[int] | int = 5,
        num_filters: int = 64,
    ):
        self.count = 0
        super().__init__(
            in_channels=in_channels,
            spatial_dims=2,
            # number of channels stay constant throughout the convolution layers
            channels=[num_filters, num_filters, num_filters],
            norm=("instance", {"num_features": in_channels}),
            kernel_size=kernel_size,
        )
        self.se_layer = se_layer if se_layer is not None else nn.Identity()
        self.dropout_layer = dropout_layer if dropout_layer is not None else nn.Identity()

    def _get_layer(self, in_channels, out_channels, dilation):
        """
        After ever convolutional layer the output is concatenated with the input and the layer before.
        The concatenated output is used as input to the next convolutional layer.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        kernelsize = self.kernel_size if self.count < 2 else (1, 1)
        # padding = None if self.count < 2 else (0, 0)
        self.count += 1
        conv = Convolution(
            spatial_dims=self.spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=1,
            kernel_size=kernelsize,
            act=self.act,
            norm=("instance", {"num_features": in_channels}),
        )
        return nn.Sequential(conv.get_submodule("adn"), conv.get_submodule("conv"))

    def forward(self, input, _):  # type: ignore[override]
        i = 0
        result = input
        result1 = input  # this will not stay this value, needed here for pylint/mypy

        for l in self.children():
            # ignoring the max (un-)pool and droupout already added in the initial initialization step
            if isinstance(l, (nn.MaxPool2d, nn.MaxUnpool2d, nn.Dropout2d)):
                continue
            # first convolutional forward
            result = l(result)
            if i == 0:
                result1 = result
                # concatenation with the input feature map
                result = torch.cat((input, result), dim=1)

            if i == 1:
                # concatenation with input feature map and feature map from first convolution
                result = torch.cat((result1, result, input), dim=1)
            i = i + 1

        # if SELayer or Dropout layer defined put output through layer before returning,
        # else it just goes through nn.Identity and the output does not change
        result = self.se_layer(result)
        result = self.dropout_layer(result)

        return result, None


class Encoder(ConvConcatDenseBlock):
    """
    Returns a convolution dense block for the encoding (down) part of a layer of the network.
    This Encoder block downpools the data with max_pool.
    Its output is used as input to the next layer down.
    New feature: it returns the indices of the max_pool to the decoder (up) path
    at the same layer to upsample the input.

    Args:
        in_channels: number of input channels.
        max_pool: predefined max_pool layer to downsample the data.
        se_layer: Squeeze and Excite block to be included, defaults to None.
        dropout: Dropout block to be included, defaults to None.
        kernel_size : kernel size of the convolutional layers. Defaults to 5*5
        num_filters : number of input channels to each convolution block. Defaults to 64
    """

    def __init__(self, in_channels: int, max_pool, se_layer, dropout, kernel_size, num_filters):
        super().__init__(in_channels, se_layer, dropout, kernel_size, num_filters)
        self.max_pool = max_pool

    def forward(self, input, indices=None):  # type: ignore[override]
        input, indices = self.max_pool(input)

        out_block, _ = super().forward(input, None)
        # safe the indices for unpool on decoder side
        return out_block, indices


class Decoder(ConvConcatDenseBlock):
    """
    Returns a convolution dense block for the decoding (up) part of a layer of the network.
    This will upsample data with an unpool block before the forward.
    It uses the indices from corresponding encoder on it's level.
    Its output is used as input to the next layer up.

    Args:
        in_channels: number of input channels.
        un_pool: predefined unpool block.
        se_layer: predefined SELayer. Defaults to None.
        dropout: predefined dropout block. Defaults to None.
        kernel_size: Kernel size of convolution layers. Defaults to 5*5.
        num_filters: number of input channels to each convolution layer. Defaults to 64.
    """

    def __init__(self, in_channels: int, un_pool, se_layer, dropout, kernel_size, num_filters):
        super().__init__(in_channels, se_layer, dropout, kernel_size, num_filters)
        self.un_pool = un_pool

    def forward(self, input, indices):  # type: ignore[override]
        out_block, _ = super().forward(input, None)
        out_block = self.un_pool(out_block, indices)
        return out_block, None


class Bottleneck(ConvConcatDenseBlock):
    """
    Returns the bottom or bottleneck layer at the bottom of a network linking encoder to decoder halves.
    It consists of a 5 * 5 convolutional layer and a batch normalization layer to separate
    the encoder and decoder part of the network, restricting information flow between the encoder and decoder.

    Args:
        in_channels: number of input channels.
        se_layer: predefined SELayer. Defaults to None.
        dropout: predefined dropout block. Defaults to None.
        un_pool: predefined unpool block.
        max_pool: predefined maxpool block.
        kernel_size: Kernel size of convolution layers. Defaults to 5*5.
        num_filters: number of input channels to each convolution layer. Defaults to 64.
    """

    def __init__(self, in_channels: int, se_layer, dropout, max_pool, un_pool, kernel_size, num_filters):
        super().__init__(in_channels, se_layer, dropout, kernel_size, num_filters)
        self.max_pool = max_pool
        self.un_pool = un_pool

    def forward(self, input, indices):  # type: ignore[override]
        out_block, indices = self.max_pool(input)
        out_block, _ = super().forward(out_block, None)
        out_block = self.un_pool(out_block, indices)
        return out_block, None


class Quicknat(nn.Module):
    """
    Model for "Quick segmentation of NeuroAnaTomy (QuickNAT) based on a deep fully convolutional neural network.
    Refer to: "QuickNAT: A Fully Convolutional Network for Quick and Accurate Segmentation of Neuroanatomy by
    Abhijit Guha Roya, Sailesh Conjetib, Nassir Navabb, Christian Wachingera"

    QuickNAT has an encoder/decoder like 2D F-CNN architecture with 4 encoders and 4 decoders separated by a bottleneck layer.
    The final layer is a classifier block with softmax.
    The architecture includes skip connections between all encoder and decoder blocks of the same spatial resolution,
    similar to the U-Net architecture.
    All Encoder and Decoder consist of three convolutional layers all with a Batch Normalization and ReLU.
    The first two convolutional layers are followed by a concatenation layer that concatenates
    the input feature map with outputs of the current and previous convolutional blocks.
    The kernel size of the first two convolutional layers is 5*5, the third convolutional layer has a kernel size of 1*1.

    Data in the encode path is downsampled using max pooling layers instead of upsamling like UNet and in the decode path
    upsampled using max un-pooling layers instead of transpose convolutions.
    The pooling is done at the beginning of the block and the unpool afterwards.
    The indices of the max pooling in the Encoder are forwarded through the layer to be available to the corresponding Decoder.

    The bottleneck block consists of a 5 * 5 convolutional layer and a batch normalization layer
    to separate the encoder and decoder part of the network,
    restricting information flow between the encoder and decoder.

    The output feature map from the last decoder block is passed to the classifier block,
    which is a convolutional layer with 1 * 1 kernel size that maps the input to an N channel feature map,
    where N is the number of segmentation classes.

    To further explain this consider the first example network given below. This network has 3 layers with strides
    of 2 for each of the middle layers (the last layer is the bottom connection which does not down/up sample). Input
    data to this network is immediately reduced in the spatial dimensions by a factor of 2 by the first convolution of
    the residual unit defining the first layer of the encode part. The last layer of the decode part will upsample its
    input (data from the previous layer concatenated with data from the skip connection) in the first convolution. this
    ensures the final output of the network has the same shape as the input.

    The original QuickNAT implementation included a `enable_test_dropout()` mechanism for uncertainty estimation during
    testing. As the dropout layers are the only stochastic components of this network calling the train() method instead
    of eval() in testing or inference has the same effect.

    Args:
        num_classes: number of classes to segmentate (output channels).
        num_channels: number of input channels.
        num_filters: number of output channels for each convolutional layer in a Dense Block.
        kernel_size: size of the kernel of each convolutional layer in a Dense Block.
        kernel_c: convolution kernel size of classifier block kernel.
        stride_convolution: convolution stride. Defaults to 1.
        pool: kernel size of the pooling layer,
        stride_pool: stride for the pooling layer.
        se_block: Squeeze and Excite block type to be included, defaults to None. Valid options : NONE, CSE, SSE, CSSE,
        droup_out: dropout ratio. Defaults to no dropout.
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        adn_ordering: a string representing the ordering of activation (A), normalization (N), and dropout (D).
            Defaults to "NA". See also: :py:class:`monai.networks.blocks.ADN`.

    Examples::

        from monai.networks.nets import QuickNAT

        # network with max pooling by a factor of 2 at each layer with no se_block.
        net = QuickNAT(
            num_classes=3,
            num_channels=1,
            num_filters=64,
            pool = 2,
            se_block = "None"
        )

    """

    def __init__(
        self,
        num_classes: int = 33,
        num_channels: int = 1,
        num_filters: int = 64,
        kernel_size: Sequence[int] | int = 5,
        kernel_c: int = 1,
        stride_conv: int = 1,
        pool: int = 2,
        stride_pool: int = 2,
        # Valid options : NONE, CSE, SSE, CSSE
        se_block: str = "None",
        drop_out: float = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        adn_ordering: str = "NA",
    ) -> None:
        self.act = act
        self.norm = norm
        self.adn_ordering = adn_ordering
        super().__init__()
        se_layer = self.get_selayer(num_filters, se_block)
        dropout_layer = get_dropout_layer(name=("dropout", {"p": drop_out}), dropout_dim=2)
        max_pool = get_pool_layer(
            name=("max", {"kernel_size": pool, "stride": stride_pool, "return_indices": True, "ceil_mode": True}),
            spatial_dims=2,
        )
        # for the unpooling layer there is currently no Monai implementation available, return to torch implementation
        un_pool = nn.MaxUnpool2d(kernel_size=pool, stride=stride_pool)

        # sequence of convolutional strides (like in UNet) not needed as they are always stride_conv. This defaults to 1.
        def _create_model(layer: int) -> nn.Module:
            """
            Builds the QuickNAT structure from the bottom up by recursing down to the bottelneck layer, then creating sequential
            blocks containing the decoder, a skip connection around the previous block, and the encoder.
            At the last layer a classifier block is added to the Sequential.

            Args:
                layer = inversproportional to the layers left to create
            """
            subblock: nn.Module
            if layer < 4:
                subblock = _create_model(layer + 1)

            else:
                subblock = Bottleneck(num_filters, se_layer, dropout_layer, max_pool, un_pool, kernel_size, num_filters)

            if layer == 1:
                down = ConvConcatDenseBlock(num_channels, se_layer, dropout_layer, kernel_size, num_filters)
                up = ConvConcatDenseBlock(num_filters * 2, se_layer, dropout_layer, kernel_size, num_filters)
                classifier = ClassifierBlock(2, num_filters, num_classes, stride_conv, kernel_c)
                return SequentialWithIdx(down, SkipConnectionWithIdx(subblock), up, classifier)
            else:
                up = Decoder(num_filters * 2, un_pool, se_layer, dropout_layer, kernel_size, num_filters)
                down = Encoder(num_filters, max_pool, se_layer, dropout_layer, kernel_size, num_filters)
                return SequentialWithIdx(down, SkipConnectionWithIdx(subblock), up)

        self.model = _create_model(1)

    def get_selayer(self, n_filters, se_block_type="None"):
        """
        Returns the SEBlock defined in the initialization of the QuickNAT model.

        Args:
            n_filters: encoding half of the layer
            se_block_type: defaults to None. Valid options are None, CSE, SSE, CSSE
        Returns: Appropriate SEBlock. SSE and CSSE not implemented in Monai yet.
        """
        if se_block_type == "CSE":
            return se.ChannelSELayer(2, n_filters)
        # not implemented in squeeze_and_excitation in monai use other squeeze_and_excitation import:
        elif se_block_type == "SSE" or se_block_type == "CSSE":
            # Throw error if squeeze_and_excitation is not installed
            if not flag:
                raise ImportError("Please install squeeze_and_excitation locally to use SpatialSELayer")
            if se_block_type == "SSE":
                return se1.SpatialSELayer(n_filters)
            else:
                return se1.ChannelSpatialSELayer(n_filters)
        else:
            return None

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input, _ = self.model(input, None)
        return input
