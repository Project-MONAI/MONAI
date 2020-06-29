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

from monai.networks.layers.factories import Dropout, Norm, Act, Conv, split_args
from monai.networks.layers.convutils import same_padding


class Convolution(nn.Sequential):
    """
    Constructs a convolution with optional dropout, normalization, and activation layers.
    """

    def __init__(
        self,
        dimensions,
        in_channels,
        out_channels,
        strides=1,
        kernel_size=3,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout=None,
        dilation=1,
        bias=True,
        conv_only=False,
        is_transposed=False,
    ) -> None:
        super().__init__()
        self.dimensions = dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_transposed = is_transposed

        padding = same_padding(kernel_size, dilation)
        conv_type = Conv[Conv.CONVTRANS if is_transposed else Conv.CONV, dimensions]

        # define the normalisation type and the arguments to the constructor
        norm_name, norm_args = split_args(norm)
        norm_type = Norm[norm_name, dimensions]

        # define the activation type and the arguments to the constructor
        act_name, act_args = split_args(act)
        act_type = Act[act_name]

        if dropout:
            # if dropout was specified simply as a p value, use default name and make a keyword map with the value
            if isinstance(dropout, (int, float)):
                drop_name = Dropout.DROPOUT
                drop_args = {"p": dropout}
            else:
                drop_name, drop_args = split_args(dropout)

            drop_type = Dropout[drop_name, dimensions]

        if is_transposed:
            conv = conv_type(in_channels, out_channels, kernel_size, strides, padding, strides - 1, 1, bias, dilation)
        else:
            conv = conv_type(in_channels, out_channels, kernel_size, strides, padding, dilation, bias=bias)

        self.add_module("conv", conv)

        if not conv_only:
            self.add_module("norm", norm_type(out_channels, **norm_args))
            if dropout:
                self.add_module("dropout", drop_type(**drop_args))

            self.add_module("act", act_type(**act_args))


class ResidualUnit(nn.Module):
    def __init__(
        self,
        dimensions,
        in_channels,
        out_channels,
        strides=1,
        kernel_size=3,
        subunits=2,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout=None,
        dilation=1,
        bias=True,
        last_conv_only=False,
    ) -> None:
        super().__init__()
        self.dimensions = dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential()
        self.residual = nn.Identity()

        padding = same_padding(kernel_size, dilation)
        schannels = in_channels
        sstrides = strides
        subunits = max(1, subunits)

        for su in range(subunits):
            conv_only = last_conv_only and su == (subunits - 1)
            unit = Convolution(
                dimensions,
                schannels,
                out_channels,
                sstrides,
                kernel_size,
                act,
                norm,
                dropout,
                dilation,
                bias,
                conv_only,
            )

            self.conv.add_module(f"unit{su:d}", unit)

            # after first loop set channels and strides to what they should be for subsequent units
            schannels = out_channels
            sstrides = 1

        # apply convolution to input to change number of output channels and size to match that coming from self.conv
        if np.prod(strides) != 1 or in_channels != out_channels:
            rkernel_size = kernel_size
            rpadding = padding

            if np.prod(strides) == 1:  # if only adapting number of channels a 1x1 kernel is used with no padding
                rkernel_size = 1
                rpadding = 0

            conv_type = Conv[Conv.CONV, dimensions]
            self.residual = conv_type(in_channels, out_channels, rkernel_size, strides, rpadding, bias=bias)

    def forward(self, x):
        res = self.residual(x)  # create the additive residual from x
        cx = self.conv(x)  # apply x to sequence of operations
        return cx + res  # add the residual to the output
