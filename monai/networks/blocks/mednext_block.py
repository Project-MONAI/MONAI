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
import torch.nn.functional as F


class MedNeXtBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: int = 7,
        do_res: int = True,
        norm_type: str = "group",
        n_groups: int or None = None,
        dim="3d",
        grn=False,
    ):

        super().__init__()

        self.do_res = do_res

        assert dim in ["2d", "3d"]
        self.dim = dim
        if self.dim == "2d":
            conv = nn.Conv2d
        else:
            conv = nn.Conv3d

        # First convolution layer with DepthWise Convolutions
        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels if n_groups is None else n_groups,
        )

        # Normalization Layer. GroupNorm is used by default.
        if norm_type == "group":
            self.norm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels)
        elif norm_type == "layer":
            self.norm = LayerNorm(normalized_shape=in_channels, data_format="channels_first")

        # Second convolution (Expansion) layer with Conv3D 1x1x1
        self.conv2 = conv(in_channels=in_channels, out_channels=exp_r * in_channels, kernel_size=1, stride=1, padding=0)

        # GeLU activations
        self.act = nn.GELU()

        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = conv(
            in_channels=exp_r * in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0
        )

        self.grn = grn
        if self.grn:
            if dim == "2d":
                self.grn_beta = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1), requires_grad=True)
            else:
                self.grn_beta = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1, 1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1, 1), requires_grad=True)

    def forward(self, x, dummy_tensor=None):

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
        self, in_channels, out_channels, exp_r=4, kernel_size=7, do_res=False, norm_type="group", dim="3d", grn=False
    ):

        super().__init__(
            in_channels, out_channels, exp_r, kernel_size, do_res=False, norm_type=norm_type, dim=dim, grn=grn
        )

        if dim == "2d":
            conv = nn.Conv2d
        else:
            conv = nn.Conv3d
        self.resample_do_res = do_res
        if do_res:
            self.res_conv = conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2)

        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x, dummy_tensor=None):

        x1 = super().forward(x)

        if self.resample_do_res:
            res = self.res_conv(x)
            x1 = x1 + res

        return x1


class MedNeXtUpBlock(MedNeXtBlock):

    def __init__(
        self, in_channels, out_channels, exp_r=4, kernel_size=7, do_res=False, norm_type="group", dim="3d", grn=False
    ):
        super().__init__(
            in_channels, out_channels, exp_r, kernel_size, do_res=False, norm_type=norm_type, dim=dim, grn=grn
        )

        self.resample_do_res = do_res

        self.dim = dim
        if dim == "2d":
            conv = nn.ConvTranspose2d
        else:
            conv = nn.ConvTranspose3d
        if do_res:
            self.res_conv = conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2)

        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x, dummy_tensor=None):

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


class OutBlock(nn.Module):

    def __init__(self, in_channels, n_classes, dim):
        super().__init__()

        if dim == "2d":
            conv = nn.ConvTranspose2d
        else:
            conv = nn.ConvTranspose3d
        self.conv_out = conv(in_channels, n_classes, kernel_size=1)

    def forward(self, x, dummy_tensor=None):
        return self.conv_out(x)


class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-5, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # beta
        self.bias = nn.Parameter(torch.zeros(normalized_shape))  # gamma
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x, dummy_tensor=False):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x
