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
# https://github.com/uni-medical/STU-Net
# and are used under the terms of the Apache License, Version 2.0.

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["BasicResBlock", "NearestUpsampleLayer", "STUNet", "STUNET_CONFIGS", "create_stunet"]


class BasicResBlock(nn.Module):
    """
    Residual convolution block used as the basic building unit of `STUNet`.

    Consists of ``conv1 -> norm1 -> act1 -> conv2 -> norm2`` with an optional
    1x1x1 convolution (``conv3``) on the shortcut when the number of channels
    or the resolution changes between input and output.

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: kernel size of the main convolutions (per axis), defaults to 3.
        padding: padding of the main convolutions (per axis), defaults to 1.
        stride: stride of the first convolution (used for downsampling), defaults to 1.
        use_1x1conv: whether to use a 1x1x1 convolution on the residual branch, defaults to False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
        padding: int | Sequence[int] = 1,
        stride: int | Sequence[int] = 1,
        use_1x1conv: bool = False,
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)
        if use_1x1conv:
            self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3 is not None:
            x = self.conv3(x)
        y += x
        return self.act2(y)


class NearestUpsampleLayer(nn.Module):
    """
    Upsampling layer of `STUNet`: nearest-neighbour interpolation followed by a 1x1x1 convolution.

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        scale_factor: upscaling factor per spatial axis (the pooling kernel size of the mirrored stage).
    """

    def __init__(self, in_channels: int, out_channels: int, scale_factor: Sequence[int]):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        return self.conv(x)


class STUNet(nn.Module):
    """
    STU-Net: a family of scalable and transferable 3D U-Nets for medical image segmentation,
    based on Huang et al., "STU-Net: Scalable and Transferable Medical Image Segmentation Models
    Empowered by Large-Scale Supervised Pre-training", arXiv:2304.06716 (2023).

    The architecture refines the nnU-Net convolutional blocks with residual connections so that
    network depth and width can be scaled together, replaces transposed convolutions with
    nearest-neighbour upsampling followed by 1x1x1 convolutions, and keeps the number of feature
    maps per stage configurable. The released family ranges from 14.6M (S) to 1.4B (H) parameters;
    models pre-trained on TotalSegmentator (1,204 CT volumes with 104 annotated structures) won the
    MICCAI 2023 ATLAS and SPPIN challenges.

    Module names (`conv_blocks_context`, `conv_blocks_localization`, `upsample_layers`, `seg_outputs`)
    and their sub-module names follow the reference implementation, so the officially released
    checkpoints can be loaded via :py:meth:`torch.nn.Module.load_state_dict` without key remapping.

    See also: reference implementation (Apache License 2.0) at https://github.com/uni-medical/STU-Net.

    Args:
        in_channels: number of input channels. Defaults to 1.
        out_channels: number of output segmentation classes. Defaults to 2.
        dims: number of feature maps at each stage, ``len(dims)`` must equal
            ``len(pool_op_kernel_sizes) + 1``. Defaults to the base configuration (B).
        depth: number of residual blocks at each stage, must have the same length as ``dims``.
            Defaults to one block per stage (B).
        pool_op_kernel_sizes: stride of the first convolution of each encoder stage (and the
            upscaling factor of the mirrored decoder stage), one entry of 3 ints per pooling step.
            The input spatial size must be divisible by the product of all entries. Defaults to
            the plan used for the released checkpoints.
        conv_kernel_sizes: kernel size of the convolutions at each stage, must have the same length
            as ``dims``. Defaults to 3x3x3 everywhere.
        deep_supervision: whether to return the per-stage segmentation outputs in training mode.
            Defaults to True.

    Raises:
        ValueError: when the lengths of ``dims``, ``depth``, ``pool_op_kernel_sizes`` or
            ``conv_kernel_sizes`` are inconsistent (``len(dims)`` must equal
            ``len(pool_op_kernel_sizes) + 1``, the others must match ``len(dims)``), when
            ``pool_op_kernel_sizes`` is empty, when ``depth`` has an entry below 1, or when a
            kernel size entry does not have 3 elements.

    Example::

        from monai.networks.nets import create_stunet

        # the released TotalSegmentator checkpoints use 105 classes (104 structures + background)
        net = create_stunet(variant="B", in_channels=1, out_channels=105)
        net(torch.rand(1, 1, 128, 128, 128))  # (1, 105, 128, 128, 128) in eval mode

    .. note::
        With ``deep_supervision=True`` the network returns, in training mode, a tuple of
        ``len(dims) - 1`` tensors ordered from the highest to the lowest resolution; every coarse
        output stays at its own resolution (no logit upsampling), matching the reference
        implementation. In eval mode only the highest resolution output is returned.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        dims: Sequence[int] = (32, 64, 128, 256, 512, 512),
        depth: Sequence[int] = (1, 1, 1, 1, 1, 1),
        pool_op_kernel_sizes: Sequence[Sequence[int]] = ((2, 2, 2),) * 4 + ((1, 1, 2),),
        conv_kernel_sizes: Sequence[Sequence[int]] = ((3, 3, 3),) * 6,
        deep_supervision: bool = True,
    ):
        super().__init__()
        if len(dims) != len(pool_op_kernel_sizes) + 1:
            raise ValueError(
                f"`dims` must have exactly one entry more than `pool_op_kernel_sizes`, "
                f"got {len(dims)} and {len(pool_op_kernel_sizes)}."
            )
        if len(conv_kernel_sizes) != len(dims):
            raise ValueError(
                f"`conv_kernel_sizes` must have the same length as `dims`, got {len(conv_kernel_sizes)} and {len(dims)}."
            )
        if len(depth) != len(dims):
            raise ValueError(f"`depth` must have the same length as `dims`, got {len(depth)} and {len(dims)}.")
        if len(pool_op_kernel_sizes) < 1:
            raise ValueError("`pool_op_kernel_sizes` must contain at least one pooling step.")
        if any(d < 1 for d in depth):
            raise ValueError(f"`depth` entries must be >= 1, got {tuple(depth)}.")
        if not all(len(k) == 3 for k in pool_op_kernel_sizes) or not all(len(k) == 3 for k in conv_kernel_sizes):
            raise ValueError(
                "`pool_op_kernel_sizes` and `conv_kernel_sizes` entries must have 3 elements (3D kernels)."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool_op_kernel_sizes = [list(k) for k in pool_op_kernel_sizes]
        self.conv_kernel_sizes = [list(k) for k in conv_kernel_sizes]
        self.conv_pad_sizes = [[i // 2 for i in k] for k in self.conv_kernel_sizes]
        self.deep_supervision = deep_supervision
        self.input_shape_must_be_divisible_by = tuple(
            math.prod(k[axis] for k in self.pool_op_kernel_sizes) for axis in range(3)
        )

        num_pool = len(self.pool_op_kernel_sizes)

        # encoder: the first block of every stage except the first downsamples with a strided convolution
        self.conv_blocks_context = nn.ModuleList(
            [
                nn.Sequential(
                    BasicResBlock(
                        in_channels if d == 0 else dims[d - 1],
                        dims[d],
                        self.conv_kernel_sizes[d],
                        self.conv_pad_sizes[d],
                        stride=1 if d == 0 else self.pool_op_kernel_sizes[d - 1],
                        use_1x1conv=True,
                    ),
                    *[
                        BasicResBlock(dims[d], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d])
                        for _ in range(depth[d] - 1)
                    ],
                )
                for d in range(num_pool + 1)
            ]
        )

        # decoder upsampling: nearest interpolation followed by 1x1x1 convolution
        self.upsample_layers = nn.ModuleList(
            [
                NearestUpsampleLayer(dims[-1 - u], dims[-2 - u], self.pool_op_kernel_sizes[-1 - u])
                for u in range(num_pool)
            ]
        )

        # decoder: the first block of each stage merges the upsampled features with the skip via a 1x1x1 convolution
        self.conv_blocks_localization = nn.ModuleList(
            [
                nn.Sequential(
                    BasicResBlock(
                        dims[-2 - u] * 2,
                        dims[-2 - u],
                        self.conv_kernel_sizes[-2 - u],
                        self.conv_pad_sizes[-2 - u],
                        use_1x1conv=True,
                    ),
                    *[
                        BasicResBlock(
                            dims[-2 - u], dims[-2 - u], self.conv_kernel_sizes[-2 - u], self.conv_pad_sizes[-2 - u]
                        )
                        for _ in range(depth[-2 - u] - 1)
                    ],
                )
                for u in range(num_pool)
            ]
        )

        # one segmentation head per decoder stage, at the stage's own resolution
        self.seg_outputs = nn.ModuleList(
            [nn.Conv3d(dims[-2 - ds], out_channels, kernel_size=1) for ds in range(num_pool)]
        )

        self.apply(partial(self._init_weights_he, slope=1e-2))

    @staticmethod
    def _init_weights_he(module: nn.Module, slope: float) -> None:
        if isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight, a=slope)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Args:
            x: input tensor with shape ``(batch, channel, depth, height, width)``.

        Returns:
            The highest-resolution segmentation output, or — when deep supervision is enabled
            in training mode — a tuple of per-stage outputs ordered from the highest to the
            lowest resolution.

        Raises:
            ValueError: when ``x`` is not 5D, or when a spatial dimension is not divisible by
                the product of all pooling kernel sizes.
        """
        if x.ndim != 5:
            raise ValueError(f"Expected a 5D input (batch, channel, depth, height, width), got {x.ndim}D.")
        for axis, (size, divisor) in enumerate(zip(x.shape[2:], self.input_shape_must_be_divisible_by)):
            if size % divisor != 0:
                raise ValueError(
                    f"Input spatial size must be divisible by {self.input_shape_must_be_divisible_by} "
                    f"(the product of all pooling kernel sizes); axis {axis} has size {size}."
                )

        skips = []
        seg_outputs = []
        x = self.conv_blocks_context[0](x)
        skips.append(x)
        for block in self.conv_blocks_context[1:-1]:
            x = block(x)
            skips.append(x)
        x = self.conv_blocks_context[-1](x)

        for upsample, localization, seg_output, skip in zip(
            self.upsample_layers, self.conv_blocks_localization, self.seg_outputs, reversed(skips)
        ):
            x = upsample(x)
            x = torch.cat((x, skip), dim=1)
            x = localization(x)
            seg_outputs.append(seg_output(x))

        if self.deep_supervision and self.training:
            return (seg_outputs[-1], *seg_outputs[:-1][::-1])
        return seg_outputs[-1]


# Scaling presets from the paper (2x width / 2x depth, and 3x / 3x for the largest model).
# The default `pool_op_kernel_sizes` and `conv_kernel_sizes` complete the released
# checkpoints' configuration; the resulting parameter counts with `in_channels=1` and
# `out_channels=105` (the released TotalSegmentator models: 104 structures + background)
# are 14.6M (S), 58.26M (B), 440.30M (L) and 1457.33M (H).
STUNET_CONFIGS: dict[str, dict] = {
    "S": {"dims": (16, 32, 64, 128, 256, 256), "depth": (1, 1, 1, 1, 1, 1)},
    "B": {"dims": (32, 64, 128, 256, 512, 512), "depth": (1, 1, 1, 1, 1, 1)},
    "L": {"dims": (64, 128, 256, 512, 1024, 1024), "depth": (2, 2, 2, 2, 2, 2)},
    "H": {"dims": (96, 192, 384, 768, 1536, 1536), "depth": (3, 3, 3, 3, 3, 3)},
}


def create_stunet(
    variant: str,
    in_channels: int = 1,
    out_channels: int = 2,
    deep_supervision: bool = True,
    pool_op_kernel_sizes: Sequence[Sequence[int]] = ((2, 2, 2),) * 4 + ((1, 1, 2),),
    conv_kernel_sizes: Sequence[Sequence[int]] = ((3, 3, 3),) * 6,
) -> STUNet:
    """
    Factory of `STUNet` scaling presets, following the family released with the paper.

    Args:
        variant: preset name, one of ``"S"``, ``"B"``, ``"L"``, ``"H"`` (case-insensitive).
        in_channels: number of input channels. Defaults to 1.
        out_channels: number of output segmentation classes. Defaults to 2.
        deep_supervision: whether to return the per-stage outputs in training mode. Defaults to True.
        pool_op_kernel_sizes: pooling strides per stage, see `STUNet`.
        conv_kernel_sizes: convolution kernel sizes per stage, see `STUNet`.

    Returns:
        STUNet: the configured network.

    Raises:
        ValueError: when ``variant`` is not a known preset.
    """
    key = variant.upper()
    if key not in STUNET_CONFIGS:
        raise ValueError(f"Unknown STU-Net variant {variant!r}, expected one of {sorted(STUNET_CONFIGS)}.")
    return STUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        deep_supervision=deep_supervision,
        pool_op_kernel_sizes=pool_op_kernel_sizes,
        conv_kernel_sizes=conv_kernel_sizes,
        **STUNET_CONFIGS[key],
    )
