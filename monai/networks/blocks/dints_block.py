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


from typing import Tuple, Union

import torch

from monai.networks.layers.factories import Conv
from monai.networks.layers.utils import get_act_layer, get_norm_layer

__all__ = ["FactorizedIncreaseBlock", "FactorizedReduceBlock", "P3DActiConvNormBlock", "ActiConvNormBlock"]


class FactorizedIncreaseBlock(torch.nn.Sequential):
    """
    Up-sampling the features by two using linear interpolation and convolutions.
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        spatial_dims: int = 3,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = "INSTANCE",
    ):
        """
        Args:
            in_channel: number of input channels
            out_channel: number of output channels
            spatial_dims: number of spatial dimensions
            act_name: activation layer type and arguments.
            norm_name: feature normalization type and arguments.
        """
        super().__init__()
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._spatial_dims = spatial_dims
        if self._spatial_dims not in (2, 3):
            raise ValueError("spatial_dims must be 2 or 3.")

        conv_type = Conv[Conv.CONV, self._spatial_dims]
        mode = "trilinear" if self._spatial_dims == 3 else "bilinear"
        self.add_module("up", torch.nn.Upsample(scale_factor=2, mode=mode, align_corners=True))
        self.add_module("acti", get_act_layer(name=act_name))
        self.add_module(
            "conv",
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
        )
        self.add_module(
            "norm", get_norm_layer(name=norm_name, spatial_dims=self._spatial_dims, channels=self._out_channel)
        )


class FactorizedReduceBlock(torch.nn.Module):
    """
    Down-sampling the feature by 2 using stride.
    The length along each spatial dimension must be a multiple of 2.
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        spatial_dims: int = 3,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = "INSTANCE",
    ):
        """
        Args:
            in_channel: number of input channels
            out_channel: number of output channels.
            spatial_dims: number of spatial dimensions.
            act_name: activation layer type and arguments.
            norm_name: feature normalization type and arguments.
        """
        super().__init__()
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._spatial_dims = spatial_dims
        if self._spatial_dims not in (2, 3):
            raise ValueError("spatial_dims must be 2 or 3.")

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
            out_channels=self._out_channel - self._out_channel // 2,
            kernel_size=1,
            stride=2,
            padding=0,
            groups=1,
            bias=False,
            dilation=1,
        )
        self.norm = get_norm_layer(name=norm_name, spatial_dims=self._spatial_dims, channels=self._out_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The length along each spatial dimension must be a multiple of 2.
        """
        x = self.act(x)
        if self._spatial_dims == 3:
            out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:, 1:])], dim=1)
        else:
            out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.norm(out)
        return out


class P3DActiConvNormBlock(torch.nn.Sequential):
    """
    -- (act) -- (conv) -- (norm) --
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        padding: int,
        mode: int = 0,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = "INSTANCE",
    ):
        """
        Args:
            in_channel: number of input channels.
            out_channel: number of output channels.
            kernel_size: kernel size to be expanded to 3D.
            padding: padding size to be expanded to 3D.
            mode: mode for the anisotropic kernels:

                - 0: ``(k, k, 1)``, ``(1, 1, k)``,
                - 1: ``(k, 1, k)``, ``(1, k, 1)``,
                - 2: ``(1, k, k)``. ``(k, 1, 1)``.

            act_name: activation layer type and arguments.
            norm_name: feature normalization type and arguments.
        """
        super().__init__()
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._p3dmode = int(mode)

        conv_type = Conv[Conv.CONV, 3]

        if self._p3dmode == 0:  # (k, k, 1), (1, 1, k)
            kernel_size0 = (kernel_size, kernel_size, 1)
            kernel_size1 = (1, 1, kernel_size)
            padding0 = (padding, padding, 0)
            padding1 = (0, 0, padding)
        elif self._p3dmode == 1:  # (k, 1, k), (1, k, 1)
            kernel_size0 = (kernel_size, 1, kernel_size)
            kernel_size1 = (1, kernel_size, 1)
            padding0 = (padding, 0, padding)
            padding1 = (0, padding, 0)
        elif self._p3dmode == 2:  # (1, k, k), (k, 1, 1)
            kernel_size0 = (1, kernel_size, kernel_size)
            kernel_size1 = (kernel_size, 1, 1)
            padding0 = (0, padding, padding)
            padding1 = (padding, 0, 0)
        else:
            raise ValueError("`mode` must be 0, 1, or 2.")

        self.add_module("acti", get_act_layer(name=act_name))
        self.add_module(
            "conv",
            conv_type(
                in_channels=self._in_channel,
                out_channels=self._in_channel,
                kernel_size=kernel_size0,
                stride=1,
                padding=padding0,
                groups=1,
                bias=False,
                dilation=1,
            ),
        )
        self.add_module(
            "conv_1",
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
        )
        self.add_module("norm", get_norm_layer(name=norm_name, spatial_dims=3, channels=self._out_channel))


class ActiConvNormBlock(torch.nn.Sequential):
    """
    -- (Acti) -- (Conv) -- (Norm) --
    """

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
        """
        Args:
            in_channel: number of input channels.
            out_channel: number of output channels.
            kernel_size: kernel size of the convolution.
            padding: padding size of the convolution.
            spatial_dims: number of spatial dimensions.
            act_name: activation layer type and arguments.
            norm_name: feature normalization type and arguments.
        """
        super().__init__()
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._spatial_dims = spatial_dims

        conv_type = Conv[Conv.CONV, self._spatial_dims]
        self.add_module("acti", get_act_layer(name=act_name))
        self.add_module(
            "conv",
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
        )
        self.add_module(
            "norm", get_norm_layer(name=norm_name, spatial_dims=self._spatial_dims, channels=self._out_channel)
        )
