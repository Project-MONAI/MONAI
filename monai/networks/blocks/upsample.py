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

from monai.networks.layers.factories import Conv, Pad, Pool
from monai.networks.utils import icnr_init, pixelshuffle
from monai.utils import InterpolateMode, UpsampleMode, ensure_tuple_rep, look_up_option

__all__ = ["Upsample", "UpSample", "SubpixelUpsample", "Subpixelupsample", "SubpixelUpSample"]


class UpSample(nn.Sequential):
    """
    Upsamples data by `scale_factor`.
    Supported modes are:

        - "deconv": uses a transposed convolution.
        - "deconvgroup": uses a transposed group convolution.
        - "nontrainable": uses :py:class:`torch.nn.Upsample`.
        - "pixelshuffle": uses :py:class:`monai.networks.blocks.SubpixelUpsample`.

    This operation will cause non-deterministic when ``mode`` is ``UpsampleMode.NONTRAINABLE``.
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
        size: tuple[int] | int | None = None,
        mode: UpsampleMode | str = UpsampleMode.DECONV,
        pre_conv: nn.Module | str | None = "default",
        post_conv: nn.Module | None = None,
        interp_mode: str = InterpolateMode.LINEAR,
        align_corners: bool | None = True,
        bias: bool = True,
        apply_pad_pool: bool = True,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of channels of the input image.
            out_channels: number of channels of the output image. Defaults to `in_channels`.
            scale_factor: multiplier for spatial size. Has to match input size if it is a tuple. Defaults to 2.
            kernel_size: kernel size used during transposed convolutions. Defaults to `scale_factor`.
            size: spatial size of the output image.
                Only used when ``mode`` is ``UpsampleMode.NONTRAINABLE``.
                In torch.nn.functional.interpolate, only one of `size` or `scale_factor` should be defined,
                thus if size is defined, `scale_factor` will not be used.
                Defaults to None.
            mode: {``"deconv"``, ``"deconvgroup"``, ``"nontrainable"``, ``"pixelshuffle"``}. Defaults to ``"deconv"``.
            pre_conv: a conv block applied before upsampling. Defaults to "default".
                When ``conv_block`` is ``"default"``, one reserved conv layer will be utilized when
                Only used in the "nontrainable" or "pixelshuffle" mode.
            post_conv: a conv block applied after upsampling. Defaults to None. Only used in the "nontrainable"  mode.
            interp_mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``}
                Only used in the "nontrainable" mode.
                If ends with ``"linear"`` will use ``spatial dims`` to determine the correct interpolation.
                This corresponds to linear, bilinear, trilinear for 1D, 2D, and 3D respectively.
                The interpolation mode. Defaults to ``"linear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html
            align_corners: set the align_corners parameter of `torch.nn.Upsample`. Defaults to True.
                Only used in the "nontrainable" mode.
            bias: whether to have a bias term in the default preconv and deconv layers. Defaults to True.
            apply_pad_pool: if True the upsampled tensor is padded then average pooling is applied with a kernel the
                size of `scale_factor` with a stride of 1. See also: :py:class:`monai.networks.blocks.SubpixelUpsample`.
                Only used in the "pixelshuffle" mode.

        """
        super().__init__()
        scale_factor_ = ensure_tuple_rep(scale_factor, spatial_dims)
        up_mode = look_up_option(mode, UpsampleMode)

        if not kernel_size:
            kernel_size_ = scale_factor_
            output_padding = padding = 0
        else:
            kernel_size_ = ensure_tuple_rep(kernel_size, spatial_dims)
            padding = tuple((k - 1) // 2 for k in kernel_size_)  # type: ignore
            output_padding = tuple(s - 1 - (k - 1) % 2 for k, s in zip(kernel_size_, scale_factor_))  # type: ignore

        if up_mode == UpsampleMode.DECONV:
            if not in_channels:
                raise ValueError(f"in_channels needs to be specified in the '{mode}' mode.")
            self.add_module(
                "deconv",
                Conv[Conv.CONVTRANS, spatial_dims](
                    in_channels=in_channels,
                    out_channels=out_channels or in_channels,
                    kernel_size=kernel_size_,
                    stride=scale_factor_,
                    padding=padding,
                    output_padding=output_padding,
                    bias=bias,
                ),
            )
        elif up_mode == UpsampleMode.DECONVGROUP:
            if not in_channels:
                raise ValueError(f"in_channels needs to be specified in the '{mode}' mode.")

            if out_channels is None:
                out_channels = in_channels
            groups = out_channels if in_channels % out_channels == 0 else 1

            self.add_module(
                "deconvgroup",
                Conv[Conv.CONVTRANS, spatial_dims](
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size_,
                    stride=scale_factor_,
                    padding=padding,
                    output_padding=output_padding,
                    groups=groups,
                    bias=bias,
                ),
            )
        elif up_mode == UpsampleMode.NONTRAINABLE:
            if pre_conv == "default" and (out_channels != in_channels):  # defaults to no conv if out_chns==in_chns
                if not in_channels:
                    raise ValueError(f"in_channels needs to be specified in the '{mode}' mode.")
                self.add_module(
                    "preconv",
                    Conv[Conv.CONV, spatial_dims](
                        in_channels=in_channels, out_channels=out_channels or in_channels, kernel_size=1, bias=bias
                    ),
                )
            elif pre_conv is not None and pre_conv != "default":
                self.add_module("preconv", pre_conv)  # type: ignore
            elif pre_conv is None and (out_channels != in_channels):
                raise ValueError(
                    "in the nontrainable mode, if not setting pre_conv, out_channels should equal to in_channels."
                )

            interp_mode = InterpolateMode(interp_mode)
            linear_mode = [InterpolateMode.LINEAR, InterpolateMode.BILINEAR, InterpolateMode.TRILINEAR]
            if interp_mode in linear_mode:  # choose mode based on dimensions
                interp_mode = linear_mode[spatial_dims - 1]

            upsample = nn.Upsample(
                size=size,
                scale_factor=None if size else scale_factor_,
                mode=interp_mode.value,
                align_corners=align_corners,
            )

            self.add_module("upsample_non_trainable", upsample)
            if post_conv:
                self.add_module("postconv", post_conv)
        elif up_mode == UpsampleMode.PIXELSHUFFLE:
            self.add_module(
                "pixelshuffle",
                SubpixelUpsample(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    scale_factor=scale_factor_[0],  # isotropic
                    conv_block=pre_conv,
                    apply_pad_pool=apply_pad_pool,
                    bias=bias,
                ),
            )
        else:
            raise NotImplementedError(f"Unsupported upsampling mode {mode}.")


class SubpixelUpsample(nn.Module):
    """
    Upsample via using a subpixel CNN. This module supports 1D, 2D and 3D input images.
    The module is consisted with two parts. First of all, a convolutional layer is employed
    to increase the number of channels into: ``in_channels * (scale_factor ** dimensions)``.
    Secondly, a pixel shuffle manipulation is utilized to aggregates the feature maps from
    low resolution space and build the super resolution space.
    The first part of the module is not fixed, a sequential layers can be used to replace the
    default single layer.

    See: Shi et al., 2016, "Real-Time Single Image and Video Super-Resolution
    Using a nEfficient Sub-Pixel Convolutional Neural Network."

    See: Aitken et al., 2017, "Checkerboard artifact free sub-pixel convolution".

    The idea comes from:
    https://arxiv.org/abs/1609.05158

    The pixel shuffle mechanism refers to:
    https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html#torch.nn.PixelShuffle.
    and:
    https://github.com/pytorch/pytorch/pull/6340.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int | None,
        out_channels: int | None = None,
        scale_factor: int = 2,
        conv_block: nn.Module | str | None = "default",
        apply_pad_pool: bool = True,
        bias: bool = True,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of channels of the input image.
            out_channels: optional number of channels of the output image.
            scale_factor: multiplier for spatial size. Defaults to 2.
            conv_block: a conv block to extract feature maps before upsampling. Defaults to None.

                - When ``conv_block`` is ``"default"``, one reserved conv layer will be utilized.
                - When ``conv_block`` is an ``nn.module``,
                  please ensure the output number of channels is divisible ``(scale_factor ** dimensions)``.

            apply_pad_pool: if True the upsampled tensor is padded then average pooling is applied with a kernel the
                size of `scale_factor` with a stride of 1. This implements the nearest neighbour resize convolution
                component of subpixel convolutions described in Aitken et al.
            bias: whether to have a bias term in the default conv_block. Defaults to True.

        """
        super().__init__()

        if scale_factor <= 0:
            raise ValueError(f"The `scale_factor` multiplier must be an integer greater than 0, got {scale_factor}.")

        self.dimensions = spatial_dims
        self.scale_factor = scale_factor

        if conv_block == "default":
            out_channels = out_channels or in_channels
            if not out_channels:
                raise ValueError("in_channels need to be specified.")
            conv_out_channels = out_channels * (scale_factor**self.dimensions)
            self.conv_block = Conv[Conv.CONV, self.dimensions](
                in_channels=in_channels, out_channels=conv_out_channels, kernel_size=3, stride=1, padding=1, bias=bias
            )

            icnr_init(self.conv_block, self.scale_factor)
        elif conv_block is None:
            self.conv_block = nn.Identity()
        else:
            self.conv_block = conv_block

        self.pad_pool: nn.Module = nn.Identity()

        if apply_pad_pool:
            pool_type = Pool[Pool.AVG, self.dimensions]
            pad_type = Pad[Pad.CONSTANTPAD, self.dimensions]

            self.pad_pool = nn.Sequential(
                pad_type(padding=(self.scale_factor - 1, 0) * self.dimensions, value=0.0),
                pool_type(kernel_size=self.scale_factor, stride=1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor in shape (batch, channel, spatial_1[, spatial_2, ...).
        """
        x = self.conv_block(x)
        if x.shape[1] % (self.scale_factor**self.dimensions) != 0:
            raise ValueError(
                f"Number of channels after `conv_block` ({x.shape[1]}) must be evenly "
                "divisible by scale_factor ** dimensions "
                f"({self.scale_factor}^{self.dimensions}={self.scale_factor**self.dimensions})."
            )
        x = pixelshuffle(x, self.dimensions, self.scale_factor)
        x = self.pad_pool(x)
        return x


Upsample = UpSample
Subpixelupsample = SubpixelUpSample = SubpixelUpsample
