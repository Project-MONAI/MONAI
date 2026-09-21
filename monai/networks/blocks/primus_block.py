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
# https://github.com/MIC-DKFZ/dynamic-network-architectures
# and are used under the terms of the Apache License, Version 2.0.

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn

from monai.networks.layers.factories import Conv, Pool
from monai.networks.layers.utils import get_act_layer, get_norm_layer

__all__ = ["LayerNormNd", "ResidualBlockD", "PrimusPatchEmbed", "PrimusPatchDecode"]


class LayerNormNd(nn.Module):
    """
    LayerNorm over the channel dimension of a channel-first tensor of shape ``(B, C, *spatial)``.

    Args:
        num_channels: number of channels ``C``.
        eps: value added to the variance for numerical stability.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        shape = (1, -1) + (1,) * (x.ndim - 2)
        return self.weight.view(shape) * x + self.bias.view(shape)


class ResidualBlockD(nn.Module):
    """
    Basic ResNet-D residual block: two 3x3 convolutions, with the downsampling skip path implemented
    as average pooling followed by a 1x1 convolution (He et al., "Bag of Tricks for Image Classification with
    Convolutional Neural Networks", CVPR 2019).

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        stride: stride of the first convolution and of the skip path. The input spatial size must be divisible
            by ``stride``, otherwise the two paths have different sizes.
        conv_bias: whether the 3x3 convolutions have a bias.
        norm: feature normalization type and arguments.
        act: activation type and arguments.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        conv_bias: bool = False,
        norm: tuple | str = ("instance", {"affine": True}),
        act: tuple | str = ("leakyrelu", {"inplace": True}),
    ) -> None:
        super().__init__()
        self.stride = stride
        conv_type = Conv[Conv.CONV, spatial_dims]
        self.conv1 = conv_type(in_channels, out_channels, 3, stride=stride, padding=1, bias=conv_bias)
        self.norm1 = get_norm_layer(norm, spatial_dims=spatial_dims, channels=out_channels)
        self.act1 = get_act_layer(act)
        self.conv2 = conv_type(out_channels, out_channels, 3, stride=1, padding=1, bias=conv_bias)
        self.norm2 = get_norm_layer(norm, spatial_dims=spatial_dims, channels=out_channels)
        self.act2 = get_act_layer(act)

        self.skip = nn.Sequential()
        if stride != 1:
            self.skip.add_module("pool", Pool[Pool.AVG, spatial_dims](stride, stride))
        if in_channels != out_channels:
            self.skip.add_module("conv", conv_type(in_channels, out_channels, 1, bias=False))
            self.skip.add_module("norm", get_norm_layer(norm, spatial_dims=spatial_dims, channels=out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if any(s % self.stride != 0 for s in x.shape[2:]):
            raise ValueError(f"input spatial size {tuple(x.shape[2:])} must be divisible by stride {self.stride}.")
        out = self.norm2(self.conv2(self.act1(self.norm1(self.conv1(x)))))
        return self.act2(out + self.skip(x))


class PrimusPatchEmbed(nn.Module):
    """
    Convolutional patch embedding of PrimusV3: a residual stem followed by one stride-2 residual stage
    per level, so the output token grid is downsampled by ``2 ** len(depth_per_level)`` along every axis.
    The input spatial size must be divisible by ``2 ** len(depth_per_level)``.
    Optionally, the features of every level but the last are projected onto the token grid and added
    with learnable scales initialized near zero.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        embed_dim: output (token) dimension.
        depth_per_level: number of residual blocks at each downsampling level.
        channels_per_level: channels of the stem followed by the channels of each level,
            of length ``len(depth_per_level) + 1``.
        add_skips: whether to add the projected features of the stem and intermediate levels to the tokens.
        norm: feature normalization type and arguments.
        act: activation type and arguments.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        embed_dim: int,
        depth_per_level: Sequence[int] = (1, 1, 1),
        channels_per_level: Sequence[int] = (32, 64, 256, 1024),
        add_skips: bool = True,
        norm: tuple | str = ("instance", {"affine": True}),
        act: tuple | str = ("leakyrelu", {"inplace": True}),
    ) -> None:
        super().__init__()
        num_levels = len(depth_per_level)
        if len(channels_per_level) != num_levels + 1:
            raise ValueError(
                f"channels_per_level must have len(depth_per_level) + 1 = {num_levels + 1} entries, "
                f"got {len(channels_per_level)}."
            )
        if any(d < 1 for d in depth_per_level):
            raise ValueError(f"depth_per_level must be positive, got {depth_per_level}.")
        self.add_skips = add_skips
        conv_type = Conv[Conv.CONV, spatial_dims]

        self.stem = ResidualBlockD(spatial_dims, in_channels, channels_per_level[0], conv_bias=True, norm=norm, act=act)
        self.stages = nn.ModuleList()
        for i in range(num_levels):
            blocks = [
                ResidualBlockD(
                    spatial_dims,
                    channels_per_level[i] if j == 0 else channels_per_level[i + 1],
                    channels_per_level[i + 1],
                    stride=2 if j == 0 else 1,
                    norm=norm,
                    act=act,
                )
                for j in range(depth_per_level[i])
            ]
            self.stages.append(nn.Sequential(*blocks))
        self.final_proj = conv_type(channels_per_level[-1], embed_dim, 1)

        if add_skips:
            # stem is at full resolution, stage i is at 1 / 2 ** (i + 1)
            self.proj_to_tokens = nn.ModuleList(
                conv_type(channels_per_level[i], embed_dim, 2 ** (num_levels - i), stride=2 ** (num_levels - i))
                for i in range(num_levels)
            )
            self.scale_proj_to_tokens = nn.ParameterList(
                nn.Parameter(torch.tensor(1e-5)) for _ in range(len(self.proj_to_tokens))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_size = 2 ** len(self.stages)
        if any(s % patch_size != 0 for s in x.shape[2:]):
            raise ValueError(f"input spatial size {tuple(x.shape[2:])} must be divisible by {patch_size}.")
        x = self.stem(x)
        skips = self.scale_proj_to_tokens[0] * self.proj_to_tokens[0](x) if self.add_skips else None
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if skips is not None and i < len(self.stages) - 1:
                skips = skips + self.scale_proj_to_tokens[i + 1] * self.proj_to_tokens[i + 1](x)
        x = self.final_proj(x)
        return x + skips if skips is not None else x


class PrimusPatchDecode(nn.Module):
    """
    Lightweight patch decoder of Primus: a stack of stride-2 transposed convolutions, each followed by a
    channel LayerNorm and an activation (except the last), that restores the input resolution from the token grid.
    The channels are reduced geometrically from ``embed_dim`` to ``out_channels``.

    Args:
        spatial_dims: number of spatial dimensions.
        patch_size: downsampling factor of the token grid along each axis; every entry must be a power of 2.
        embed_dim: token dimension.
        out_channels: number of output channels.
        act: activation type and arguments.
    """

    def __init__(
        self, spatial_dims: int, patch_size: Sequence[int], embed_dim: int, out_channels: int, act: tuple | str = "gelu"
    ) -> None:
        super().__init__()
        if len(patch_size) != spatial_dims:
            raise ValueError(f"patch_size must have {spatial_dims} entries, got {patch_size}.")
        if any(p < 2 or p & (p - 1) for p in patch_size):
            raise ValueError(f"patch_size entries must be powers of 2 greater than 1, got {patch_size}.")

        num_stages = int(math.log2(max(patch_size)))
        # stride 2 along an axis until its patch size is reached, coarsest stages first
        strides = [[2 if (p / 2**n) % 2 == 0 else 1 for p in patch_size] for n in range(num_stages)][::-1]
        dim_red = (embed_dim / (2 * out_channels)) ** (1 / num_stages)
        channels = [embed_dim] + [
            max(8, round((embed_dim / dim_red ** (i + 1) + 1e-6) / 8) * 8) for i in range(num_stages)
        ]
        channels[-1] = out_channels

        convt_type = Conv[Conv.CONVTRANS, spatial_dims]
        stages: list[nn.Module] = [
            nn.Sequential(
                convt_type(channels[s], channels[s + 1], kernel_size=strides[s], stride=strides[s]),
                LayerNormNd(channels[s + 1]),
                get_act_layer(act),
            )
            for s in range(num_stages - 1)
        ]
        stages.append(convt_type(channels[-2], channels[-1], kernel_size=strides[-1], stride=strides[-1]))
        self.decode = nn.Sequential(*stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(x)
