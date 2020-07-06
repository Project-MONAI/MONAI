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

from typing import Optional, Union

import torch
import torch.nn as nn

from monai.networks.layers.factories import Conv
from monai.utils import UpsampleMode


class UpSample(nn.Module):
    """
    Upsample with either kernel 1 conv + interpolation or transposed conv.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: Optional[int] = None,
        scale_factor=2,
        with_conv: bool = False,
        mode: Union[UpsampleMode, str] = UpsampleMode.LINEAR,
        align_corners: Optional[bool] = True,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of channels of the input image.
            out_channels: number of channels of the output image. Defaults to `in_channels`.
            scale_factor: multiplier for spatial size. Has to match input size if it is a tuple. Defaults to 2.
            with_conv: whether to use a transposed convolution for upsampling. Defaults to False.
            mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``}
                If ends with ``"linear"`` will use ``spatial dims`` to determine the correct interpolation.
                This corresponds to linear, bilinear, trilinear for 1D, 2D, and 3D respectively.
                The interpolation mode. Defaults to ``"linear"``.
                See also: https://pytorch.org/docs/stable/nn.html#upsample
            align_corners: set the align_corners parameter of `torch.nn.Upsample`. Defaults to True.
        """
        super().__init__()
        if not out_channels:
            out_channels = in_channels
        if not with_conv:
            mode = UpsampleMode(mode)
            linear_mode = [UpsampleMode.LINEAR, UpsampleMode.BILINEAR, UpsampleMode.TRILINEAR]
            if mode in linear_mode:  # choose mode based on spatial_dims
                mode = linear_mode[spatial_dims - 1]
            self.upsample = nn.Sequential(
                Conv[Conv.CONV, spatial_dims](in_channels=in_channels, out_channels=out_channels, kernel_size=1),
                nn.Upsample(scale_factor=scale_factor, mode=mode.value, align_corners=align_corners),
            )
        else:
            self.upsample = Conv[Conv.CONVTRANS, spatial_dims](
                in_channels=in_channels, out_channels=out_channels, kernel_size=scale_factor, stride=scale_factor
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor in shape (batch, channel, spatial_1[, spatial_2, ...).
        """
        return self.upsample(x)
