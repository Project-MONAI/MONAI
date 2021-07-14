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

from typing import Sequence, Union

import numpy as np
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, UnetUpBlock, get_conv_layer
from monai.networks.layers.factories import Norm
from monai.networks.layers.utils import get_act_layer


class _UnetResBlockV1(UnetResBlock):
    """
    UnetResBlock for backward compatibility purpose.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: str,
    ):
        nn.Module.__init__(self)
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            conv_only=True,
        )
        self.conv2 = get_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            conv_only=True,
        )
        self.conv3 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            conv_only=True,
        )
        self.lrelu = get_act_layer(("leakyrelu", {"inplace": True, "negative_slope": 0.01}))
        self.norm1 = _get_norm_layer(spatial_dims, out_channels, norm_name)
        self.norm2 = _get_norm_layer(spatial_dims, out_channels, norm_name)
        self.norm3 = _get_norm_layer(spatial_dims, out_channels, norm_name)
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True


class _UnetBasicBlockV1(UnetBasicBlock):
    """
    UnetBasicBlock for backward compatibility purpose.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: str,
    ):
        nn.Module.__init__(self)
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            conv_only=True,
        )
        self.conv2 = get_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            conv_only=True,
        )
        self.lrelu = get_act_layer(("leakyrelu", {"inplace": True, "negative_slope": 0.01}))
        self.norm1 = _get_norm_layer(spatial_dims, out_channels, norm_name)
        self.norm2 = _get_norm_layer(spatial_dims, out_channels, norm_name)


class _UnetUpBlockV1(UnetUpBlock):
    """
    UnetUpBlock for backward compatibility purpose.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: str,
    ):
        nn.Module.__init__(self)
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )
        self.conv_block = _UnetBasicBlockV1(
            spatial_dims,
            out_channels + out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
        )


def _get_norm_layer(spatial_dims: int, out_channels: int, norm_name: str, num_groups: int = 16):
    if norm_name not in ["batch", "instance", "group"]:
        raise ValueError(f"Unsupported normalization mode: {norm_name}")
    if norm_name != "group":
        return Norm[norm_name, spatial_dims](out_channels, affine=True)
    if out_channels % num_groups != 0:
        raise AssertionError("out_channels should be divisible by num_groups.")
    return Norm[norm_name, spatial_dims](num_groups=num_groups, num_channels=out_channels, affine=True)
