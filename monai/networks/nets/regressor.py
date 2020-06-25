# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import torch.nn as nn

from monai.networks.layers.factories import Norm, Act
from monai.networks.blocks import Convolution, ResidualUnit
from monai.networks.layers.simplelayers import Reshape
from monai.networks.layers.convutils import same_padding, calculate_out_shape


class Regressor(nn.Module):
    """
    This defines a network for relating large-sized input tensors to small output tensors, ie. regressing large
    values to a prediction. An output of a single dimension can be used as value regression or multi-label
    classification prediction, an output of a single value can be used as a discriminator or critic prediction.
    """

    def __init__(
        self,
        in_shape,
        out_shape,
        channels,
        strides,
        kernel_size=3,
        num_res_units=2,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout=None,
        bias=True,
    ):
        """
        Construct the regressor network with the number of layers defined by `channels` and `strides`. Inputs are
        first passed through the convolutional layers in the forward pass, the output from this is then pass
        through a fully connected layer to relate them to the final output tensor.

        Args:
            in_shape: tuple of integers stating the dimension of the input tensor (minus batch dimension)
            out_shape: tuple of integers stating the dimension of the final output tensor
            channels: tuple of integers stating the output channels of each convolutional layer
            strides: tuple of integers stating the stride (downscale factor) of each convolutional layer
            kernel_size: integer or tuple of integers stating size of convolutional kernels
            num_res_units: integer stating number of convolutions in residual units, 0 means no residual units
            act: name or type defining activation layers
            norm: name or type defining normalization layers
            dropout: optional float value in range [0, 1] stating dropout probability for layers, None for no dropout
            bias: boolean stating if convolution layers should have a bias component
        """
        super().__init__()

        self.in_channels, *self.in_shape = in_shape
        self.dimensions = len(self.in_shape)
        self.channels = channels
        self.strides = strides
        self.out_shape = out_shape
        self.kernel_size = kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.net = nn.Sequential()

        echannel = self.in_channels

        padding = same_padding(kernel_size)

        self.final_size = np.asarray(self.in_shape, np.int)
        self.reshape = Reshape(*self.out_shape)

        # encode stage
        for i, (c, s) in enumerate(zip(self.channels, self.strides)):
            layer = self._get_layer(echannel, c, s, i == len(channels) - 1)
            echannel = c  # use the output channel number as the input for the next loop
            self.net.add_module("layer_%i" % i, layer)
            self.final_size = calculate_out_shape(self.final_size, kernel_size, s, padding)

        self.final = self._get_final_layer((echannel,) + self.final_size)

    def _get_layer(self, in_channels, out_channels, strides, is_last):
        """
        Returns a layer accepting inputs with `in_channels` number of channels and producing outputs of `out_channels`
        number of channels. The `strides` indicates downsampling factor, ie. convolutional stride. If `is_last`
        is True this is the final layer and is not expected to include activation and normalization layers.
        """

        common_kwargs = dict(
            dimensions=self.dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
        )

        if self.num_res_units > 0:
            layer = ResidualUnit(subunits=self.num_res_units, last_conv_only=is_last, **common_kwargs)
        else:
            layer = Convolution(conv_only=is_last, **common_kwargs)

        return layer

    def _get_final_layer(self, in_shape):
        linear = nn.Linear(int(np.product(in_shape)), int(np.product(self.out_shape)))
        return nn.Sequential(nn.Flatten(), linear)

    def forward(self, x):
        x = self.net(x)
        x = self.final(x)
        x = self.reshape(x)
        return x
