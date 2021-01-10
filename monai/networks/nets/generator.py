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
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import Reshape
from monai.utils import ensure_tuple, ensure_tuple_rep


class Generator(nn.Module):
    """
    Defines a simple generator network accepting a latent vector and through a sequence of convolution layers
    constructs an output tensor of greater size and high dimensionality. The method `_get_layer` is used to
    create each of these layers, override this method to define layers beyond the default Convolution or
    ResidualUnit layers.

    For example, a generator accepting a latent vector if shape (42,24) and producing an output volume of
    shape (1,64,64) can be constructed as:

        gen = Generator((42, 24), (64, 8, 8), (32, 16, 1), (2, 2, 2))
    """

    def __init__(
        self,
        latent_shape: Sequence[int],
        start_shape: Sequence[int],
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 2,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout: Optional[float] = None,
        bias: bool = True,
    ) -> None:
        """
        Construct the generator network with the number of layers defined by `channels` and `strides`. In the
        forward pass a `nn.Linear` layer relates the input latent vector to a tensor of dimensions `start_shape`,
        this is then fed forward through the sequence of convolutional layers. The number of layers is defined by
        the length of `channels` and `strides` which must match, each layer having the number of output channels
        given in `channels` and an upsample factor given in `strides` (ie. a transpose convolution with that stride
        size).

        Args:
            latent_shape: tuple of integers stating the dimension of the input latent vector (minus batch dimension)
            start_shape: tuple of integers stating the dimension of the tensor to pass to convolution subnetwork
            channels: tuple of integers stating the output channels of each convolutional layer
            strides: tuple of integers stating the stride (upscale factor) of each convolutional layer
            kernel_size: integer or tuple of integers stating size of convolutional kernels
            num_res_units: integer stating number of convolutions in residual units, 0 means no residual units
            act: name or type defining activation layers
            norm: name or type defining normalization layers
            dropout: optional float value in range [0, 1] stating dropout probability for layers, None for no dropout
            bias: boolean stating if convolution layers should have a bias component
        """
        super().__init__()

        self.in_channels, *self.start_shape = ensure_tuple(start_shape)
        self.dimensions = len(self.start_shape)

        self.latent_shape = ensure_tuple(latent_shape)
        self.channels = ensure_tuple(channels)
        self.strides = ensure_tuple(strides)
        self.kernel_size = ensure_tuple_rep(kernel_size, self.dimensions)
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(int(np.prod(self.latent_shape)), int(np.prod(start_shape)))
        self.reshape = Reshape(*start_shape)
        self.conv = nn.Sequential()

        echannel = self.in_channels

        # transform tensor of shape `start_shape' into output shape through transposed convolutions and residual units
        for i, (c, s) in enumerate(zip(channels, strides)):
            is_last = i == len(channels) - 1
            layer = self._get_layer(echannel, c, s, is_last)
            self.conv.add_module("layer_%i" % i, layer)
            echannel = c

    def _get_layer(
        self, in_channels: int, out_channels: int, strides: int, is_last: bool
    ) -> Union[Convolution, nn.Sequential]:
        """
        Returns a layer accepting inputs with `in_channels` number of channels and producing outputs of `out_channels`
        number of channels. The `strides` indicates upsampling factor, ie. transpose convolutional stride. If `is_last`
        is True this is the final layer and is not expected to include activation and normalization layers.
        """

        layer: Union[Convolution, nn.Sequential]

        layer = Convolution(
            in_channels=in_channels,
            strides=strides,
            is_transposed=True,
            conv_only=is_last or self.num_res_units > 0,
            dimensions=self.dimensions,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                in_channels=out_channels,
                subunits=self.num_res_units,
                last_conv_only=is_last,
                dimensions=self.dimensions,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
            )

            layer = nn.Sequential(layer, ru)

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.linear(x)
        x = self.reshape(x)
        x = self.conv(x)
        return x
