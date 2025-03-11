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

from collections.abc import Sequence

import torch
import torch.nn as nn

from monai.networks.layers.factories import Conv, Pool
from monai.networks.utils import pixelunshuffle
from monai.utils import DownsampleMode, ensure_tuple_rep, look_up_option

__all__ = ["MaxAvgPool", "DownSample", "Downsample", "SubpixelDownsample", "SubpixelDownSample", "Subpixeldownsample"]


class MaxAvgPool(nn.Module):
    """
    Downsample with both maxpooling and avgpooling,
    double the channel size by concatenating the downsampled feature maps.
    """

    def __init__(
        self,
        spatial_dims: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int | None = None,
        padding: Sequence[int] | int = 0,
        ceil_mode: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            kernel_size: the kernel size of both pooling operations.
            stride: the stride of the window. Default value is `kernel_size`.
            padding: implicit zero padding to be added to both pooling operations.
            ceil_mode: when True, will use ceil instead of floor to compute the output shape.
        """
        super().__init__()
        _params = {
            "kernel_size": ensure_tuple_rep(kernel_size, spatial_dims),
            "stride": None if stride is None else ensure_tuple_rep(stride, spatial_dims),
            "padding": ensure_tuple_rep(padding, spatial_dims),
            "ceil_mode": ceil_mode,
        }
        self.max_pool = Pool[Pool.MAX, spatial_dims](**_params)
        self.avg_pool = Pool[Pool.AVG, spatial_dims](**_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor in shape (batch, channel, spatial_1[, spatial_2, ...]).

        Returns:
            Tensor in shape (batch, 2*channel, spatial_1[, spatial_2, ...]).
        """
        return torch.cat([self.max_pool(x), self.avg_pool(x)], dim=1)


class DownSample(nn.Sequential):
    """
    Downsamples data by `scale_factor`.

    Supported modes are:

    - "conv": uses a strided convolution for learnable downsampling.
    - "convgroup": uses a grouped strided convolution for efficient feature reduction.
    - "nontrainable": uses :py:class:`torch.nn.Upsample` with inverse scale factor.
    - "pixelunshuffle": uses :py:class:`monai.networks.blocks.PixelUnshuffle` for channel-space rearrangement.

    This operation will cause non-deterministic behavior when ``mode`` is ``DownsampleMode.NONTRAINABLE``.
    Please check the link below for more details:
    https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms

    This module can optionally take a pre-convolution
    (often used to map the number of features from `in_channels` to `out_channels`).
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int | None = None,
        out_channels: int | None = None,
        scale_factor: Sequence[float] | float = 2,
        kernel_size: Sequence[float] | float | None = None,
        mode: DownsampleMode | str = DownsampleMode.CONV,
        pre_conv: nn.Module | str | None = "default",
        post_conv: nn.Module | None = None,
        bias: bool = True,
    ) -> None:
        """
        Downsamples data by `scale_factor`.
        Supported modes are:

            - DownsampleMode.CONV: uses a strided convolution for learnable downsampling.
            - DownsampleMode.CONVGROUP: uses a grouped strided convolution for efficient feature reduction.
            - DownsampleMode.MAXPOOL: uses maxpooling for non-learnable downsampling.
            - DownsampleMode.AVGPOOL: uses average pooling for non-learnable downsampling.
            - DownsampleMode.PIXELUNSHUFFLE: uses :py:class:`monai.networks.blocks.SubpixelDownsample`.

        This operation will cause non-deterministic behavior when ``mode`` is ``DownsampleMode.NONTRAINABLE``.
        Please check the link below for more details:
        https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms

        This module can optionally take a pre-convolution and post-convolution
        (often used to map the number of features from `in_channels` to `out_channels`).

        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of channels of the input image.
            out_channels: number of channels of the output image. Defaults to `in_channels`.
            scale_factor: multiplier for spatial size reduction. Has to match input size if it is a tuple. Defaults to 2.
            kernel_size: kernel size used during convolutions. Defaults to `scale_factor`.
            mode: {``DownsampleMode.CONV``, ``DownsampleMode.CONVGROUP``, ``DownsampleMode.MAXPOOL``, ``DownsampleMode.AVGPOOL``,
                ``DownsampleMode.PIXELUNSHUFFLE``}. Defaults to ``DownsampleMode.CONV``.
            pre_conv: a conv block applied before downsampling. Defaults to "default".
                When ``conv_block`` is ``"default"``, one reserved conv layer will be utilized.
                Only used in the "maxpool", "avgpool" or "pixelunshuffle" modes.
            post_conv: a conv block applied after downsampling. Defaults to None. Only used in the "maxpool" and "avgpool" modes.
            bias: whether to have a bias term in the default preconv and conv layers. Defaults to True.
        """
        super().__init__()

        scale_factor_ = ensure_tuple_rep(scale_factor, spatial_dims)
        down_mode = look_up_option(mode, DownsampleMode)

        if not kernel_size:
            kernel_size_ = scale_factor_
            padding = ensure_tuple_rep(0, spatial_dims)
        else:
            kernel_size_ = ensure_tuple_rep(kernel_size, spatial_dims)
            padding = tuple((k - 1) // 2 for k in kernel_size_)

        if down_mode == DownsampleMode.CONV:
            if not in_channels:
                raise ValueError("in_channels needs to be specified in conv mode")
            self.add_module(
                "conv",
                Conv[Conv.CONV, spatial_dims](
                    in_channels=in_channels,
                    out_channels=out_channels or in_channels,
                    kernel_size=kernel_size_,
                    stride=scale_factor_,
                    padding=padding,
                    bias=bias,
                ),
            )
        elif down_mode == DownsampleMode.CONVGROUP:
            if not in_channels:
                raise ValueError("in_channels needs to be specified")
            if out_channels is None:
                out_channels = in_channels
            groups = in_channels if out_channels % in_channels == 0 else 1
            self.add_module(
                "convgroup",
                Conv[Conv.CONV, spatial_dims](
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size_,
                    stride=scale_factor_,
                    padding=padding,
                    groups=groups,
                    bias=bias,
                ),
            )
        elif down_mode == DownsampleMode.MAXPOOL:
            if pre_conv == "default" and (out_channels != in_channels):
                if not in_channels:
                    raise ValueError("in_channels needs to be specified")
                self.add_module(
                    "preconv",
                    Conv[Conv.CONV, spatial_dims](
                        in_channels=in_channels, out_channels=out_channels or in_channels, kernel_size=1, bias=bias
                    ),
                )
            self.add_module(
                "maxpool", Pool[Pool.MAX, spatial_dims](kernel_size=kernel_size_, stride=scale_factor_, padding=padding)
            )
            if post_conv:
                self.add_module("postconv", post_conv)

        elif down_mode == DownsampleMode.AVGPOOL:
            if pre_conv == "default" and (out_channels != in_channels):
                if not in_channels:
                    raise ValueError("in_channels needs to be specified")
                self.add_module(
                    "preconv",
                    Conv[Conv.CONV, spatial_dims](
                        in_channels=in_channels, out_channels=out_channels or in_channels, kernel_size=1, bias=bias
                    ),
                )
            self.add_module(
                "avgpool", Pool[Pool.AVG, spatial_dims](kernel_size=kernel_size_, stride=scale_factor_, padding=padding)
            )
            if post_conv:
                self.add_module("postconv", post_conv)

        elif down_mode == DownsampleMode.PIXELUNSHUFFLE:
            self.add_module(
                "pixelunshuffle",
                SubpixelDownsample(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    scale_factor=scale_factor_[0],
                    conv_block=pre_conv,
                    bias=bias,
                ),
            )


class SubpixelDownsample(nn.Module):
    """
    Downsample via using a subpixel CNN. This module supports 1D, 2D and 3D input images.
    The module consists of two parts. First, a convolutional layer is employed
    to adjust the number of channels. Secondly, a pixel unshuffle manipulation
    rearranges the spatial information into channel space, effectively reducing
    spatial dimensions while increasing channel depth.

    The pixel unshuffle operation is the inverse of pixel shuffle, rearranging dimensions
    from (B, C, H*r, W*r) to (B, C*r², H, W) for 2D images or from (B, C, H*r, W*r, D*r) to (B, C*r³, H, W, D) in 3D case.

    Example: (1, 1, 4, 4) with r=2 becomes (1, 4, 2, 2).

    See: Shi et al., 2016, "Real-Time Single Image and Video Super-Resolution
    Using a nEfficient Sub-Pixel Convolutional Neural Network."

    The pixel unshuffle mechanism is the inverse operation of:
    https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/blocks/upsample.py
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int | None,
        out_channels: int | None = None,
        scale_factor: int = 2,
        conv_block: nn.Module | str | None = "default",
        bias: bool = True,
    ) -> None:
        """
        Downsamples data by rearranging spatial information into channel space.
        This reduces spatial dimensions while increasing channel depth.

        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of channels of the input image.
            out_channels: optional number of channels of the output image.
            scale_factor: factor to reduce the spatial dimensions by. Defaults to 2.
            conv_block: a conv block to adjust channels before downsampling. Defaults to None.
                When ``conv_block`` is ``"default"``, one reserved conv layer will be utilized.
                When ``conv_block`` is an ``nn.module``,
                please ensure the input number of channels matches requirements.
            bias: whether to have a bias term in the default conv_block. Defaults to True.
        """
        super().__init__()

        if scale_factor <= 0:
            raise ValueError(f"The `scale_factor` multiplier must be an integer greater than 0, got {scale_factor}.")

        self.dimensions = spatial_dims
        self.scale_factor = scale_factor

        if conv_block == "default":
            if not in_channels:
                raise ValueError("in_channels need to be specified.")
            out_channels = out_channels or in_channels
            self.conv_block = Conv[Conv.CONV, self.dimensions](
                in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=bias
            )
        elif conv_block is None:
            self.conv_block = nn.Identity()
        else:
            self.conv_block = conv_block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor in shape (batch, channel, spatial_1[, spatial_2, ...).
        Returns:
            Tensor with reduced spatial dimensions and increased channel depth.
        """
        x = self.conv_block(x)
        if not all(d % self.scale_factor == 0 for d in x.shape[2:]):
            raise ValueError(
                f"All spatial dimensions {x.shape[2:]} must be evenly " f"divisible by scale_factor {self.scale_factor}"
            )
        x = pixelunshuffle(x, self.dimensions, self.scale_factor)
        return x


Downsample = DownSample
SubpixelDownSample = Subpixeldownsample = SubpixelDownsample
