# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from monai.networks.blocks import Convolution, ResidualUnit
from monai.networks.layers.convutils import calculate_out_shape, same_padding
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import Reshape
from monai.utils import ensure_tuple, ensure_tuple_rep

__all__ = ["Regressor"]


class Regressor(nn.Module):
    """
    This defines a network for relating large-sized input tensors to small output tensors, ie. regressing large
    values to a prediction. An output of a single dimension can be used as value regression or multi-label
    classification prediction, an output of a single value can be used as a discriminator or critic prediction.

    The network is constructed as a sequence of layers, either :py:class:`monai.networks.blocks.Convolution` or
    :py:class:`monai.networks.blocks.ResidualUnit`, with a final fully-connected layer resizing the output from the
    blocks to the final size. Each block is defined with a stride value typically used to downsample the input using
    strided convolutions. In this way each block progressively condenses information from the input into a deep
    representation the final fully-connected layer relates to a final result.

    Args:
        in_shape: tuple of integers stating the dimension of the input tensor (minus batch dimension)
        out_shape: tuple of integers stating the dimension of the final output tensor (minus batch dimension)
        channels: tuple of integers stating the output channels of each convolutional layer
        strides: tuple of integers stating the stride (downscale factor) of each convolutional layer
        kernel_size: integer or tuple of integers stating size of convolutional kernels
        num_res_units: integer stating number of convolutions in residual units, 0 means no residual units
        act: name or type defining activation layers
        norm: name or type defining normalization layers
        dropout: optional float value in range [0, 1] stating dropout probability for layers, None for no dropout
        bias: boolean stating if convolution layers should have a bias component

    Examples::

        # infers a 2-value result (eg. a 2D cartesian coordinate) from a 64x64 image
        net = Regressor((1, 64, 64), (2,), (2, 4, 8), (2, 2, 2))

    """

    def __init__(
        self,
        in_shape: Sequence[int],
        out_shape: Sequence[int],
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 2,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout: Optional[float] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels, *self.in_shape = ensure_tuple(in_shape)
        self.dimensions = len(self.in_shape)
        self.channels = ensure_tuple(channels)
        self.strides = ensure_tuple(strides)
        self.out_shape = ensure_tuple(out_shape)
        self.kernel_size = ensure_tuple_rep(kernel_size, self.dimensions)
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.net = nn.Sequential()

        echannel = self.in_channels

        padding = same_padding(kernel_size)

        self.final_size = np.asarray(self.in_shape, dtype=int)
        self.reshape = Reshape(*self.out_shape)

        # encode stage
        for i, (c, s) in enumerate(zip(self.channels, self.strides)):
            layer = self._get_layer(echannel, c, s, i == len(channels) - 1)
            echannel = c  # use the output channel number as the input for the next loop
            self.net.add_module("layer_%i" % i, layer)
            self.final_size = calculate_out_shape(self.final_size, kernel_size, s, padding)  # type: ignore

        self.final = self._get_final_layer((echannel,) + self.final_size)

    def _get_layer(
        self, in_channels: int, out_channels: int, strides: int, is_last: bool
    ) -> Union[ResidualUnit, Convolution]:
        """
        Returns a layer accepting inputs with `in_channels` number of channels and producing outputs of `out_channels`
        number of channels. The `strides` indicates downsampling factor, ie. convolutional stride. If `is_last`
        is True this is the final layer and is not expected to include activation and normalization layers.
        """

        layer: Union[ResidualUnit, Convolution]

        if self.num_res_units > 0:
            layer = ResidualUnit(
                subunits=self.num_res_units,
                last_conv_only=is_last,
                spatial_dims=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
            )
        else:
            layer = Convolution(
                conv_only=is_last,
                spatial_dims=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
            )

        return layer

    def _get_final_layer(self, in_shape: Sequence[int]):
        linear = nn.Linear(int(np.product(in_shape)), int(np.product(self.out_shape)))
        return nn.Sequential(nn.Flatten(), linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = self.final(x)
        x = self.reshape(x)
        return x
