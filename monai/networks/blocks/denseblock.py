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

from typing import Sequence

import torch
import torch.nn as nn

from monai.networks.blocks import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm

__ALL__ = ["DenseBlock", "ConvDenseBlock", "ConvConcatDenseBlock", "Bottleneck", "Encoder", "Decoder"]


class DenseBlock(nn.Sequential):
    """
    A DenseBlock is a sequence of layers where each layer's outputs are concatenated with their inputs. This has the
    effect of accumulating outputs from previous layers as inputs to later ones and as the final output of the block.

    Args:
        layers: sequence of nn.Module objects to define the individual layers of the dense block
    """

    def __init__(self, layers: Sequence[nn.Module]):
        super().__init__()
        for i, l in enumerate(layers):
            self.add_module(f"layers{i}", l)

    def forward(self, x):
        for l in self.children():
            result = l(x)
            x = torch.cat([x, result], 1)

        return x


class ConvDenseBlock(DenseBlock):
    """
    This dense block is defined as a sequence of `Convolution` or `ResidualUnit` blocks. The `_get_layer` method returns
    an object for each layer and can be overridden to change the composition of the block.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        channels: output channels for each layer.
        dilations: dilation value for each layer.
        kernel_size: convolution kernel size. Defaults to 3.
        num_res_units: number of convolutions. Defaults to 2.
        adn_ordering: a string representing the ordering of activation, normalization, and dropout. Defaults to "NDA".
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        bias: whether to have a bias term. Defaults to True.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        channels: Sequence[int],
        dilations: Sequence[int] | None = None,
        kernel_size: Sequence[int] | int = 3,
        num_res_units: int = 0,
        adn_ordering: str = "NDA",
        act: tuple | str | None = Act.PRELU,
        norm: tuple | str | None = Norm.INSTANCE,
        dropout: int | None = None,
        bias: bool = True,
    ):
        self.spatial_dims = spatial_dims
        self.kernel_size = kernel_size
        self.num_res_units = num_res_units
        self.adn_ordering = adn_ordering
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias

        l_channels = in_channels
        dilations = dilations if dilations is not None else ([1] * len(channels))
        layers = []

        if len(channels) != len(dilations):
            raise ValueError("Length of `channels` and `dilations` must match")

        for c, d in zip(channels, dilations):
            layer = self._get_layer(l_channels, c, d)
            layers.append(layer)
            l_channels += c

        super().__init__(layers)

    def _get_layer(self, in_channels, out_channels, dilation):
        if self.num_res_units > 0:
            return ResidualUnit(
                spatial_dims=self.spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                adn_ordering=self.adn_ordering,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                dilation=dilation,
                bias=self.bias,
            )
        else:
            return Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                dilation=dilation,
                bias=self.bias,
            )


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
        se_layer: nn.Module | None = nn.Identity,
        dropout_layer: type[nn.Dropout2d] | None = nn.Identity,
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
        self.se_layer = se_layer
        self.dropout_layer = dropout_layer

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

    def forward(self, input, _):
        i = 0
        result = input
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

        # if SELayer or Dropout layer defined put output through layer before returning, else it just goes through nn.Identity and the output does not change
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

    def forward(self, input, indices=None):
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

    def forward(self, input, indices):
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

    def forward(self, input, indices):
        out_block, indices = self.max_pool(input)
        out_block, _ = super().forward(out_block, None)
        out_block = self.un_pool(out_block, indices)
        return out_block, None
