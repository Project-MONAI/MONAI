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


from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.layers.factories import Conv
from monai.networks.layers.utils import get_act_layer, get_norm_layer

__all__ = ["FactorizedIncreaseBlock", "FactorizedReduceBlock", "P3DReLUConvNormBlock", "ReLUConvNormBlock"]


class FactorizedIncreaseBlock(nn.Module):
    """
    Up-sampling the feature by 2 using stride.
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        spatial_dims: int = 3,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = "INSTANCE",
    ):
        super().__init__()
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._spatial_dims = spatial_dims

        conv_type = Conv[Conv.CONV, self._spatial_dims]

        self.op = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True),
            get_act_layer(name=act_name),
            conv_type(
                in_channels=self._in_channel,
                out_channels=self._out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bias=False,
                dilation=1,
            ),
            get_norm_layer(name=norm_name, spatial_dims=self._spatial_dims, channels=self._out_channel),
        )

    def forward(self, x):
        return self.op(x)


class FactorizedReduceBlock(nn.Module):
    """
    Down-sampling the feature by 2 using stride.
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        spatial_dims: int = 3,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = "INSTANCE",
    ):
        super().__init__()
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._spatial_dims = spatial_dims

        conv_type = Conv[Conv.CONV, self._spatial_dims]

        self.act = get_act_layer(name=act_name)
        self.conv_1 = conv_type(
            in_channels=self._in_channel,
            out_channels=self._out_channel // 2,
            kernel_size=1,
            stride=2,
            padding=0,
            groups=1,
            bias=False,
            dilation=1,
        )
        self.conv_2 = conv_type(
            in_channels=self._in_channel,
            out_channels=self._out_channel // 2,
            kernel_size=1,
            stride=2,
            padding=0,
            groups=1,
            bias=False,
            dilation=1,
        )
        self.norm = get_norm_layer(name=norm_name, spatial_dims=self._spatial_dims, channels=self._out_channel)

    def forward(self, x):
        x = self.act(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:, 1:])], dim=1)
        out = self.norm(out)
        return out


class P3DReLUConvNormBlock(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        padding: int,
        p3dmode: int = 0,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = "INSTANCE",
    ):
        super().__init__()
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._p3dmode = p3dmode

        conv_type = Conv[Conv.CONV, 3]

        if self._p3dmode == 0:  # 3 x 3 x 1
            kernel_size0 = (kernel_size, kernel_size, 1)
            kernel_size1 = (1, 1, kernel_size)
            padding0 = (padding, padding, 0)
            padding1 = (0, 0, padding)
        elif self._p3dmode == 1:  # 3 x 1 x 3
            kernel_size0 = (kernel_size, 1, kernel_size)
            kernel_size1 = (1, kernel_size, 1)
            padding0 = (padding, 0, padding)
            padding1 = (0, padding, 0)
        elif self._p3dmode == 2:  # 1 x 3 x 3
            kernel_size0 = (1, kernel_size, kernel_size)
            kernel_size1 = (kernel_size, 1, 1)
            padding0 = (0, padding, padding)
            padding1 = (padding, 0, 0)

        self.op = nn.Sequential(
            get_act_layer(name=act_name),
            conv_type(
                in_channels=self._in_channel,
                out_channels=self._out_channel,
                kernel_size=kernel_size0,
                stride=1,
                padding=padding0,
                groups=1,
                bias=False,
                dilation=1,
            ),
            conv_type(
                in_channels=self._in_channel,
                out_channels=self._out_channel,
                kernel_size=kernel_size1,
                stride=1,
                padding=padding1,
                groups=1,
                bias=False,
                dilation=1,
            ),
            get_norm_layer(name=norm_name, spatial_dims=3, channels=self._out_channel),
        )

    def forward(self, x):
        return self.op(x)


class ReLUConvNormBlock(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int = 3,
        padding: int = 1,
        spatial_dims: int = 3,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = "INSTANCE",
    ):
        super().__init__()
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._spatial_dims = spatial_dims

        conv_type = Conv[Conv.CONV, self._spatial_dims]

        self.op = nn.Sequential(
            get_act_layer(name=act_name),
            conv_type(
                in_channels=self._in_channel,
                out_channels=self._out_channel,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=1,
                bias=False,
                dilation=1,
            ),
            get_norm_layer(name=norm_name, spatial_dims=self._spatial_dims, channels=self._out_channel),
        )

    def forward(self, x):
        return self.op(x)
