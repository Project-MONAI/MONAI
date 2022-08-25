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

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm

__ALL__ = ["DenseBlock", "ConvDenseBlock"]


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
        dilations: Optional[Sequence[int]] = None,
        kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        adn_ordering: str = "NDA",
        act: Optional[Union[Tuple, str]] = Act.PRELU,
        norm: Optional[Union[Tuple, str]] = Norm.INSTANCE,
        dropout: Optional[int] = None,
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
