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

import copy
from collections.abc import Callable
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from monai.networks.blocks.upsample import UpSample
from monai.networks.layers.factories import Act, Conv, Norm, split_args
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode, has_option

__all__ = ["SegResNetDS", "SegResNetDS2"]


def scales_for_resolution(resolution: tuple | list, n_stages: int | None = None):
    """
    A helper function to compute a schedule of scale at different downsampling levels,
    given the input resolution.

    .. code-block:: python

        scales_for_resolution(resolution=[1,1,5], n_stages=5)

    Args:
        resolution: input image resolution (in mm)
        n_stages: optionally the number of stages of the network
    """

    ndim = len(resolution)
    res = np.array(resolution)
    if not all(res > 0):
        raise ValueError("Resolution must be positive")

    nl = np.floor(np.log2(np.max(res) / res)).astype(np.int32)
    scales = [tuple(np.where(2**i >= 2**nl, 1, 2)) for i in range(max(nl))]
    if n_stages and n_stages > max(nl):
        scales = scales + [(2,) * ndim] * (n_stages - max(nl))
    else:
        scales = scales[:n_stages]
    return scales


def aniso_kernel(scale: tuple | list):
    """
    A helper function to compute kernel_size, padding and stride for the given scale

    Args:
        scale: scale from a current scale level
    """
    kernel_size = [3 if scale[k] > 1 else 1 for k in range(len(scale))]
    padding = [k // 2 for k in kernel_size]
    return kernel_size, padding, scale


class SegResBlock(nn.Module):
    """
    Residual network block used SegResNet based on `3D MRI brain tumor segmentation using autoencoder regularization
    <https://arxiv.org/pdf/1810.11654.pdf>`_.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm: tuple | str,
        kernel_size: tuple | int = 3,
        act: tuple | str = "relu",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            norm: feature normalization type and arguments.
            kernel_size: convolution kernel size. Defaults to 3.
            act: activation type and arguments. Defaults to ``RELU``.
        """
        super().__init__()

        if isinstance(kernel_size, (tuple, list)):
            padding = tuple(k // 2 for k in kernel_size)
        else:
            padding = kernel_size // 2  # type: ignore

        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act1 = get_act_layer(act)
        self.conv1 = Conv[Conv.CONV, spatial_dims](
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )

        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act2 = get_act_layer(act)
        self.conv2 = Conv[Conv.CONV, spatial_dims](
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )

    def forward(self, x):
        identity = x
        x = self.conv2(self.act2(self.norm2(self.conv1(self.act1(self.norm1(x))))))
        x += identity
        return x


class SegResEncoder(nn.Module):
    """
    SegResEncoder based on the encoder structure in `3D MRI brain tumor segmentation using autoencoder regularization
    <https://arxiv.org/pdf/1810.11654.pdf>`_.

    Args:
        spatial_dims: spatial dimension of the input data. Defaults to 3.
        init_filters: number of output channels for initial convolution layer. Defaults to 32.
        in_channels: number of input channels for the network. Defaults to 1.
        out_channels: number of output channels for the network. Defaults to 2.
        act: activation type and arguments. Defaults to ``RELU``.
        norm: feature normalization type and arguments. Defaults to ``BATCH``.
        blocks_down: number of downsample blocks in each layer. Defaults to ``[1,2,2,4]``.
        head_module: optional callable module to apply to the final features.
        anisotropic_scales: optional list of scale for each scale level.
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 32,
        in_channels: int = 1,
        act: tuple | str = "relu",
        norm: tuple | str = "batch",
        blocks_down: tuple = (1, 2, 2, 4),
        head_module: nn.Module | None = None,
        anisotropic_scales: tuple | None = None,
    ):
        super().__init__()

        if spatial_dims not in (1, 2, 3):
            raise ValueError("`spatial_dims` can only be 1, 2 or 3.")

        # ensure normalization has affine trainable parameters (if not specified)
        norm = split_args(norm)
        if has_option(Norm[norm[0], spatial_dims], "affine"):
            norm[1].setdefault("affine", True)  # type: ignore

        # ensure activation is inplace (if not specified)
        act = split_args(act)
        if has_option(Act[act[0]], "inplace"):
            act[1].setdefault("inplace", True)  # type: ignore

        filters = init_filters  # base number of features

        kernel_size, padding, _ = aniso_kernel(anisotropic_scales[0]) if anisotropic_scales else (3, 1, 1)
        self.conv_init = Conv[Conv.CONV, spatial_dims](
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=kernel_size,
            padding=padding,
            stride=1,
            bias=False,
        )
        self.layers = nn.ModuleList()

        for i in range(len(blocks_down)):
            level = nn.ModuleDict()

            kernel_size, padding, stride = aniso_kernel(anisotropic_scales[i]) if anisotropic_scales else (3, 1, 2)
            blocks = [
                SegResBlock(spatial_dims=spatial_dims, in_channels=filters, kernel_size=kernel_size, norm=norm, act=act)
                for _ in range(blocks_down[i])
            ]
            level["blocks"] = nn.Sequential(*blocks)

            if i < len(blocks_down) - 1:
                level["downsample"] = Conv[Conv.CONV, spatial_dims](
                    in_channels=filters,
                    out_channels=2 * filters,
                    bias=False,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            else:
                level["downsample"] = nn.Identity()

            self.layers.append(level)
            filters *= 2

        self.head_module = head_module
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.init_filters = init_filters
        self.norm = norm
        self.act = act
        self.spatial_dims = spatial_dims

    def _forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        x = self.conv_init(x)

        for level in self.layers:
            x = level["blocks"](x)
            outputs.append(x)
            x = level["downsample"](x)

        if self.head_module is not None:
            outputs = self.head_module(outputs)

        return outputs

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self._forward(x)


class SegResNetDS(nn.Module):
    """
    SegResNetDS based on `3D MRI brain tumor segmentation using autoencoder regularization
    <https://arxiv.org/pdf/1810.11654.pdf>`_.
    It is similar to https://docs.monai.io/en/stable/networks.html#segresnet, with several
    improvements including deep supervision and non-isotropic kernel support.

    Args:
        spatial_dims: spatial dimension of the input data. Defaults to 3.
        init_filters: number of output channels for initial convolution layer. Defaults to 32.
        in_channels: number of input channels for the network. Defaults to 1.
        out_channels: number of output channels for the network. Defaults to 2.
        act: activation type and arguments. Defaults to ``RELU``.
        norm: feature normalization type and arguments. Defaults to ``BATCH``.
        blocks_down: number of downsample blocks in each layer. Defaults to ``[1,2,2,4]``.
        blocks_up: number of upsample blocks (optional).
        dsdepth: number of levels for deep supervision. This will be the length of the list of outputs at each scale level.
                 At dsdepth==1,only a single output is returned.
        preprocess: optional callable function to apply before the model's forward pass
        resolution: optional input image resolution. When provided, the network will first use non-isotropic kernels to bring
                    image spacing into an approximately isotropic space.
                    Otherwise, by default, the kernel size and downsampling is always isotropic.

    """

    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 32,
        in_channels: int = 1,
        out_channels: int = 2,
        act: tuple | str = "relu",
        norm: tuple | str = "batch",
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple | None = None,
        dsdepth: int = 1,
        preprocess: nn.Module | Callable | None = None,
        upsample_mode: UpsampleMode | str = "deconv",
        resolution: tuple | None = None,
    ):
        super().__init__()

        if spatial_dims not in (1, 2, 3):
            raise ValueError("`spatial_dims` can only be 1, 2 or 3.")

        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.norm = norm
        self.blocks_down = blocks_down
        self.dsdepth = max(dsdepth, 1)
        self.resolution = resolution
        self.preprocess = preprocess

        if resolution is not None:
            if not isinstance(resolution, (list, tuple)):
                raise TypeError("resolution must be a tuple")
            elif not all(r > 0 for r in resolution):
                raise ValueError("resolution must be positive")

        # ensure normalization had affine trainable parameters (if not specified)
        norm = split_args(norm)
        if has_option(Norm[norm[0], spatial_dims], "affine"):
            norm[1].setdefault("affine", True)  # type: ignore

        # ensure activation is inplace (if not specified)
        act = split_args(act)
        if has_option(Act[act[0]], "inplace"):
            act[1].setdefault("inplace", True)  # type: ignore

        anisotropic_scales = None
        if resolution:
            anisotropic_scales = scales_for_resolution(resolution, n_stages=len(blocks_down))
        self.anisotropic_scales = anisotropic_scales

        self.encoder = SegResEncoder(
            spatial_dims=spatial_dims,
            init_filters=init_filters,
            in_channels=in_channels,
            act=act,
            norm=norm,
            blocks_down=blocks_down,
            anisotropic_scales=anisotropic_scales,
        )

        n_up = len(blocks_down) - 1
        if blocks_up is None:
            blocks_up = (1,) * n_up  # assume 1 upsample block per level
        self.blocks_up = blocks_up

        filters = init_filters * 2**n_up
        self.up_layers = nn.ModuleList()

        for i in range(n_up):
            filters = filters // 2
            kernel_size, _, stride = (
                aniso_kernel(anisotropic_scales[len(blocks_up) - i - 1]) if anisotropic_scales else (3, 1, 2)
            )

            level = nn.ModuleDict()
            level["upsample"] = UpSample(
                mode=upsample_mode,
                spatial_dims=spatial_dims,
                in_channels=2 * filters,
                out_channels=filters,
                kernel_size=kernel_size,
                scale_factor=stride,
                bias=False,
                align_corners=False,
            )
            blocks = [
                SegResBlock(spatial_dims=spatial_dims, in_channels=filters, kernel_size=kernel_size, norm=norm, act=act)
                for _ in range(blocks_up[i])
            ]
            level["blocks"] = nn.Sequential(*blocks)

            if len(blocks_up) - i <= dsdepth:  # deep supervision heads
                level["head"] = Conv[Conv.CONV, spatial_dims](
                    in_channels=filters, out_channels=out_channels, kernel_size=1, bias=True
                )
            else:
                level["head"] = nn.Identity()

            self.up_layers.append(level)

        if n_up == 0:  # in a corner case of flat structure (no downsampling), attache a single head
            level = nn.ModuleDict(
                {
                    "upsample": nn.Identity(),
                    "blocks": nn.Identity(),
                    "head": Conv[Conv.CONV, spatial_dims](
                        in_channels=filters, out_channels=out_channels, kernel_size=1, bias=True
                    ),
                }
            )
            self.up_layers.append(level)

    def shape_factor(self):
        """
        Calculate the factors (divisors) that the input image shape must be divisible by
        """
        if self.anisotropic_scales is None:
            d = [2 ** (len(self.blocks_down) - 1)] * self.spatial_dims
        else:
            d = list(np.prod(np.array(self.anisotropic_scales[:-1]), axis=0))
        return d

    def is_valid_shape(self, x):
        """
        Calculate if the input shape is divisible by the minimum factors for the current network configuration
        """
        a = [i % j == 0 for i, j in zip(x.shape[2:], self.shape_factor())]
        return all(a)

    def _forward(self, x: torch.Tensor) -> Union[None, torch.Tensor, list[torch.Tensor]]:
        if self.preprocess is not None:
            x = self.preprocess(x)

        if not self.is_valid_shape(x):
            raise ValueError(f"Input spatial dims {x.shape} must be divisible by {self.shape_factor()}")

        x_down = self.encoder(x)

        x_down.reverse()
        x = x_down.pop(0)

        if len(x_down) == 0:
            x_down = [torch.zeros(1, device=x.device, dtype=x.dtype)]

        outputs: list[torch.Tensor] = []

        i = 0
        for level in self.up_layers:
            x = level["upsample"](x)
            x += x_down.pop(0)
            x = level["blocks"](x)

            if len(self.up_layers) - i <= self.dsdepth:
                outputs.append(level["head"](x))
            i = i + 1

        outputs.reverse()

        # in eval() mode, always return a single final output
        if not self.training or len(outputs) == 1:
            return outputs[0]

        # return a list of DS outputs
        return outputs

    def forward(self, x: torch.Tensor) -> Union[None, torch.Tensor, list[torch.Tensor]]:
        return self._forward(x)


class SegResNetDS2(SegResNetDS):
    """
    SegResNetDS2 adds an additional decorder branch to SegResNetDS and is the image encoder of VISTA3D
     <https://arxiv.org/abs/2406.05285>`_.

    Args:
        spatial_dims: spatial dimension of the input data. Defaults to 3.
        init_filters: number of output channels for initial convolution layer. Defaults to 32.
        in_channels: number of input channels for the network. Defaults to 1.
        out_channels: number of output channels for the network. Defaults to 2.
        act: activation type and arguments. Defaults to ``RELU``.
        norm: feature normalization type and arguments. Defaults to ``BATCH``.
        blocks_down: number of downsample blocks in each layer. Defaults to ``[1,2,2,4]``.
        blocks_up: number of upsample blocks (optional).
        dsdepth: number of levels for deep supervision. This will be the length of the list of outputs at each scale level.
                 At dsdepth==1,only a single output is returned.
        preprocess: optional callable function to apply before the model's forward pass
        resolution: optional input image resolution. When provided, the network will first use non-isotropic kernels to bring
                    image spacing into an approximately isotropic space.
                    Otherwise, by default, the kernel size and downsampling is always isotropic.

    """

    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 32,
        in_channels: int = 1,
        out_channels: int = 2,
        act: tuple | str = "relu",
        norm: tuple | str = "batch",
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple | None = None,
        dsdepth: int = 1,
        preprocess: nn.Module | Callable | None = None,
        upsample_mode: UpsampleMode | str = "deconv",
        resolution: tuple | None = None,
    ):
        super().__init__(
            spatial_dims=spatial_dims,
            init_filters=init_filters,
            in_channels=in_channels,
            out_channels=out_channels,
            act=act,
            norm=norm,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            dsdepth=dsdepth,
            preprocess=preprocess,
            upsample_mode=upsample_mode,
            resolution=resolution,
        )

        self.up_layers_auto = nn.ModuleList([copy.deepcopy(layer) for layer in self.up_layers])

    def forward(  # type: ignore
        self, x: torch.Tensor, with_point: bool = True, with_label: bool = True
    ) -> tuple[Union[None, torch.Tensor, list[torch.Tensor]], Union[None, torch.Tensor, list[torch.Tensor]]]:
        """
        Args:
            x: input tensor.
            with_point: if true, return the point branch output.
            with_label: if true, return the label branch output.
        """
        if self.preprocess is not None:
            x = self.preprocess(x)

        if not self.is_valid_shape(x):
            raise ValueError(f"Input spatial dims {x.shape} must be divisible by {self.shape_factor()}")

        x_down = self.encoder(x)

        x_down.reverse()
        x = x_down.pop(0)

        if len(x_down) == 0:
            x_down = [torch.zeros(1, device=x.device, dtype=x.dtype)]

        outputs: list[torch.Tensor] = []
        outputs_auto: list[torch.Tensor] = []
        x_ = x.clone()
        if with_point:
            i = 0
            for level in self.up_layers:
                x = level["upsample"](x)
                x = x + x_down[i]
                x = level["blocks"](x)

                if len(self.up_layers) - i <= self.dsdepth:
                    outputs.append(level["head"](x))
                i = i + 1

            outputs.reverse()
        x = x_
        if with_label:
            i = 0
            for level in self.up_layers_auto:
                x = level["upsample"](x)
                x = x + x_down[i]
                x = level["blocks"](x)

                if len(self.up_layers) - i <= self.dsdepth:
                    outputs_auto.append(level["head"](x))
                i = i + 1

            outputs_auto.reverse()

        return outputs[0] if len(outputs) == 1 else outputs, outputs_auto[0] if len(outputs_auto) == 1 else outputs_auto

    def set_auto_grad(self, auto_freeze=False, point_freeze=False):
        """
        Args:
            auto_freeze: if true, freeze the image encoder and the auto-branch.
            point_freeze: if true, freeze the image encoder and the point-branch.
        """
        for param in self.encoder.parameters():
            param.requires_grad = (not auto_freeze) and (not point_freeze)

        for param in self.up_layers_auto.parameters():
            param.requires_grad = not auto_freeze

        for param in self.up_layers.parameters():
            param.requires_grad = not point_freeze
