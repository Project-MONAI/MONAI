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

# Portions of this code are derived from the original repository at:
# https://github.com/MIC-DKFZ/MedNeXt
# and are used under the terms of the Apache License, Version 2.0.

from __future__ import annotations

import torch
import torch.nn as nn

all = ["MedNeXtBlock", "MedNeXtDownBlock", "MedNeXtUpBlock", "MedNeXtOutBlock"]


def get_conv_layer(spatial_dim: int = 3, transpose: bool = False):
    if spatial_dim == 2:
        return nn.ConvTranspose2d if transpose else nn.Conv2d
    else:  # spatial_dim == 3
        return nn.ConvTranspose3d if transpose else nn.Conv3d


class MedNeXtBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int = 4,
        kernel_size: int = 7,
        use_residual_connection: int = True,
        norm_type: str = "group",
        dim="3d",
        grn=False,
    ):

        super().__init__()

        self.do_res = use_residual_connection

        self.dim = dim
        conv = get_conv_layer(spatial_dim=2 if dim == "2d" else 3)
        grn_parameter_shape = (1,) * (2 if dim == "2d" else 3)
        # First convolution layer with DepthWise Convolutions
        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels,
        )

        # Normalization Layer. GroupNorm is used by default.
        if norm_type == "group":
            self.norm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels)  # type: ignore
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(
                normalized_shape=[in_channels] + [kernel_size] * (2 if dim == "2d" else 3)  # type: ignore
            )
        # Second convolution (Expansion) layer with Conv3D 1x1x1
        self.conv2 = conv(
            in_channels=in_channels, out_channels=expansion_ratio * in_channels, kernel_size=1, stride=1, padding=0
        )

        # GeLU activations
        self.act = nn.GELU()

        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = conv(
            in_channels=expansion_ratio * in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0
        )

        self.grn = grn
        if self.grn:
            grn_parameter_shape = (1, expansion_ratio * in_channels) + grn_parameter_shape
            self.grn_beta = nn.Parameter(torch.zeros(grn_parameter_shape), requires_grad=True)
            self.grn_gamma = nn.Parameter(torch.zeros(grn_parameter_shape), requires_grad=True)

    def forward(self, x):

        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))

        if self.grn:
            # gamma, beta: learnable affine transform parameters
            # X: input of shape (N,C,H,W,D)
            if self.dim == "2d":
                gx = torch.norm(x1, p=2, dim=(-2, -1), keepdim=True)
            else:
                gx = torch.norm(x1, p=2, dim=(-3, -2, -1), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
            x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1
        x1 = self.conv3(x1)
        if self.do_res:
            x1 = x + x1
        return x1


class MedNeXtDownBlock(MedNeXtBlock):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int = 4,
        kernel_size: int = 7,
        use_residual_connection: bool = False,
        norm_type: str = "group",
        dim: str = "3d",
        grn: bool = False,
    ):

        super().__init__(
            in_channels,
            out_channels,
            expansion_ratio,
            kernel_size,
            use_residual_connection=False,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        conv = get_conv_layer(spatial_dim=2 if dim == "2d" else 3)
        self.resample_do_res = use_residual_connection
        if use_residual_connection:
            self.res_conv = conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2)

        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x):

        x1 = super().forward(x)

        if self.resample_do_res:
            res = self.res_conv(x)
            x1 = x1 + res

        return x1


class MedNeXtUpBlock(MedNeXtBlock):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int = 4,
        kernel_size: int = 7,
        use_residual_connection: bool = False,
        norm_type: str = "group",
        dim: str = "3d",
        grn: bool = False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            expansion_ratio,
            kernel_size,
            use_residual_connection=False,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.resample_do_res = use_residual_connection

        self.dim = dim
        conv = get_conv_layer(spatial_dim=2 if dim == "2d" else 3, transpose=True)
        if use_residual_connection:
            self.res_conv = conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2)

        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x):

        x1 = super().forward(x)
        # Asymmetry but necessary to match shape

        if self.dim == "2d":
            x1 = torch.nn.functional.pad(x1, (1, 0, 1, 0))
        else:
            x1 = torch.nn.functional.pad(x1, (1, 0, 1, 0, 1, 0))

        if self.resample_do_res:
            res = self.res_conv(x)
            if self.dim == "2d":
                res = torch.nn.functional.pad(res, (1, 0, 1, 0))
            else:
                res = torch.nn.functional.pad(res, (1, 0, 1, 0, 1, 0))
            x1 = x1 + res

        return x1


class MedNeXtOutBlock(nn.Module):

    def __init__(self, in_channels, n_classes, dim):
        super().__init__()

        conv = get_conv_layer(spatial_dim=2 if dim == "2d" else 3, transpose=True)
        self.conv_out = conv(in_channels, n_classes, kernel_size=1)

    def forward(self, x):
        return self.conv_out(x)
