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

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

import torch
import torch.nn as nn

from monai.networks.layers.drop_path import DropPath
from monai.networks.layers.factories import Conv, Pool
from monai.networks.layers.utils import get_act_layer
from monai.networks.layers.weight_init import trunc_normal_

__all__ = [
    "ConvNeXt",
    "Convnext",
    "ConvNeXtTiny",
    "Convnext_tiny",
    "convnext_tiny",
    "ConvNeXtSmall",
    "Convnext_small",
    "convnext_small",
    "ConvNeXtBase",
    "Convnext_base",
    "convnext_base",
    "ConvNeXtLarge",
    "Convnext_large",
    "convnext_large",
    "ConvNeXtXLarge",
    "Convnext_xlarge",
    "convnext_xlarge",
]


class LayerNormNd(nn.Module):
    """
    Layer normalization over the channel dimension of a channels-first tensor.

    `torch.nn.LayerNorm` normalizes over the trailing dimensions, so it expects a channels-last layout
    such as (batch, *spatial, channel). Convolutional feature maps in MONAI are channels-first,
    (batch, channel, *spatial), and this module normalizes those over the channel dimension only,
    for any number of spatial dimensions.

    Args:
        num_channels: number of channels of the input, i.e. the size of dimension 1.
        spatial_dims: number of spatial dimensions of the input image.
        eps: value added to the denominator for numerical stability.
    """

    def __init__(self, num_channels: int, spatial_dims: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        # broadcast the affine parameters against (batch, channel, *spatial); precomputed so that the
        # module is scriptable without inspecting the rank of the input at runtime.
        self.param_shape = [1, num_channels] + [1] * spatial_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight.view(self.param_shape) * x + self.bias.view(self.param_shape)


class _ConvNeXtBlock(nn.Module):
    """
    A single ConvNeXt block: depthwise convolution, normalization, and an inverted bottleneck of two
    pointwise convolutions, added back to the input through an optional layer scale and drop path.

    The reference implementation applies the pointwise convolutions as linear layers on a permuted,
    channels-last tensor. Using 1x1 convolutions instead is equivalent, and keeps the block agnostic to
    the number of spatial dimensions.

    Args:
        spatial_dims: number of spatial dimensions of the input image.
        dim: number of channels of the input and of the output.
        kernel_size: size of the depthwise convolution kernel.
        drop_path: stochastic depth rate.
        layer_scale_init_value: initial value of the layer scale applied to the residual branch,
            no layer scale is applied when this is not positive.
        act: activation type and arguments.
    """

    def __init__(
        self,
        spatial_dims: int,
        dim: int,
        kernel_size: int = 7,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        act: str | tuple = "gelu",
    ) -> None:
        super().__init__()
        conv_type: type[nn.Conv1d | nn.Conv2d | nn.Conv3d] = Conv[Conv.CONV, spatial_dims]

        self.dwconv = conv_type(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = LayerNormNd(dim, spatial_dims=spatial_dims)
        self.pwconv1 = conv_type(dim, 4 * dim, kernel_size=1)
        self.act = get_act_layer(name=act)
        self.pwconv2 = conv_type(4 * dim, dim, kernel_size=1)
        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        else:
            self.register_parameter("gamma", None)
        self.param_shape = [1, dim] + [1] * spatial_dims
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        gamma = self.gamma
        if gamma is not None:
            x = gamma.view(self.param_shape) * x
        residual: torch.Tensor = self.drop_path(x)
        return identity + residual


class ConvNeXt(nn.Module):
    """
    ConvNeXt based on: `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_.
    Adapted from the 2D reference implementation: https://github.com/facebookresearch/ConvNeXt.

    The network is a classification backbone built from four stages of :py:class:`_ConvNeXtBlock`,
    separated by downsampling layers. Unlike the reference implementation it supports 1D, 2D and 3D
    inputs, which makes it usable for volumetric medical images.

    Each downsampling layer halves the spatial size and the patchify stem reduces it by a factor of 4,
    so every spatial dimension of the input should be divisible by 32.

    Args:
        spatial_dims: number of spatial dimensions of the input image.
        in_channels: number of the input channel.
        out_channels: number of the output classes.
        depths: number of blocks in each of the four stages.
        features: number of channels in each of the four stages.
        drop_path_rate: stochastic depth rate, increased linearly over the blocks.
        layer_scale_init_value: initial value of the layer scale applied to each residual branch,
            no layer scale is applied when this is not positive.
        kernel_size: size of the depthwise convolution kernel of each block.
        act: activation type and arguments. Defaults to gelu.

    Raises:
        ValueError: when `spatial_dims` is not one of (1, 2, 3).
        ValueError: when `depths` and `features` have different lengths.

    Example::

        # 3D ConvNeXt-Tiny for binary classification of single channel volumes
        net = ConvNeXtTiny(spatial_dims=3, in_channels=1, out_channels=2)
        output = net(torch.randn(2, 1, 32, 32, 32))  # (2, 2)
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (3, 3, 9, 3),
        features: Sequence[int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        kernel_size: int = 7,
        act: str | tuple = "gelu",
    ) -> None:
        super().__init__()

        if spatial_dims not in (1, 2, 3):
            raise ValueError(f"`spatial_dims` should be 1, 2 or 3, got {spatial_dims}.")
        if len(depths) != len(features):
            raise ValueError(
                f"`depths` and `features` should have the same length, got {len(depths)} and {len(features)}."
            )

        conv_type: type[nn.Conv1d | nn.Conv2d | nn.Conv3d] = Conv[Conv.CONV, spatial_dims]
        avg_pool_type: type[nn.AdaptiveAvgPool1d | nn.AdaptiveAvgPool2d | nn.AdaptiveAvgPool3d] = Pool[
            Pool.ADAPTIVEAVG, spatial_dims
        ]

        # stochastic depth increases linearly over the blocks of the whole network
        drop_path_rates = [float(x) for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.features = nn.Sequential(
            OrderedDict(
                [
                    (
                        "stem",
                        nn.Sequential(
                            conv_type(in_channels, features[0], kernel_size=4, stride=4),
                            LayerNormNd(features[0], spatial_dims=spatial_dims),
                        ),
                    )
                ]
            )
        )
        for i, depth in enumerate(depths):
            if i > 0:
                downsample = nn.Sequential(
                    LayerNormNd(features[i - 1], spatial_dims=spatial_dims),
                    conv_type(features[i - 1], features[i], kernel_size=2, stride=2),
                )
                self.features.add_module(f"downsample{i}", downsample)
            blocks = [
                _ConvNeXtBlock(
                    spatial_dims=spatial_dims,
                    dim=features[i],
                    kernel_size=kernel_size,
                    drop_path=drop_path_rates[sum(depths[:i]) + j],
                    layer_scale_init_value=layer_scale_init_value,
                    act=act,
                )
                for j in range(depth)
            ]
            self.features.add_module(f"stage{i + 1}", nn.Sequential(*blocks))

        # pooling and classification, the final normalization is over the pooled feature vector and so
        # it is channels-last and uses `nn.LayerNorm` directly.
        self.class_layers = nn.Sequential(
            OrderedDict(
                [
                    ("pool", avg_pool_type(1)),
                    ("flatten", nn.Flatten(1)),
                    ("norm", nn.LayerNorm(features[-1], eps=1e-6)),
                    ("out", nn.Linear(features[-1], out_channels)),
                ]
            )
        )

        for m in self.modules():
            if isinstance(m, (conv_type, nn.Linear)):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.class_layers(x)
        return x


class ConvNeXtTiny(ConvNeXt):
    """ConvNeXt-T, the tiny variant of :py:class:`ConvNeXt`."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (3, 3, 9, 3),
        features: Sequence[int] = (96, 192, 384, 768),
        **kwargs,
    ) -> None:
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            depths=depths,
            features=features,
            **kwargs,
        )


class ConvNeXtSmall(ConvNeXt):
    """ConvNeXt-S, the small variant of :py:class:`ConvNeXt`."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (3, 3, 27, 3),
        features: Sequence[int] = (96, 192, 384, 768),
        **kwargs,
    ) -> None:
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            depths=depths,
            features=features,
            **kwargs,
        )


class ConvNeXtBase(ConvNeXt):
    """ConvNeXt-B, the base variant of :py:class:`ConvNeXt`."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (3, 3, 27, 3),
        features: Sequence[int] = (128, 256, 512, 1024),
        **kwargs,
    ) -> None:
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            depths=depths,
            features=features,
            **kwargs,
        )


class ConvNeXtLarge(ConvNeXt):
    """ConvNeXt-L, the large variant of :py:class:`ConvNeXt`."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (3, 3, 27, 3),
        features: Sequence[int] = (192, 384, 768, 1536),
        **kwargs,
    ) -> None:
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            depths=depths,
            features=features,
            **kwargs,
        )


class ConvNeXtXLarge(ConvNeXt):
    """ConvNeXt-XL, the extra large variant of :py:class:`ConvNeXt`."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (3, 3, 27, 3),
        features: Sequence[int] = (256, 512, 1024, 2048),
        **kwargs,
    ) -> None:
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            depths=depths,
            features=features,
            **kwargs,
        )


Convnext = ConvNeXt
Convnext_tiny = convnext_tiny = ConvNeXtTiny
Convnext_small = convnext_small = ConvNeXtSmall
Convnext_base = convnext_base = ConvNeXtBase
Convnext_large = convnext_large = ConvNeXtLarge
Convnext_xlarge = convnext_xlarge = ConvNeXtXLarge
