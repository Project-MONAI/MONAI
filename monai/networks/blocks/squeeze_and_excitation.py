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

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks import Convolution
from monai.networks.layers.factories import Act, Conv, Norm, Pool, split_args


class ChannelSELayer(nn.Module):
    """
    Re-implementation of the Squeeze-and-Excitation block based on:
    "Hu et al., Squeeze-and-Excitation Networks, https://arxiv.org/abs/1709.01507".
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        r: int = 2,
        acti_type_1: Union[Tuple[str, Dict], str] = ("relu", {"inplace": True}),
        acti_type_2: Union[Tuple[str, Dict], str] = "sigmoid",
        add_residual: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
            in_channels: number of input channels.
            r: the reduction ratio r in the paper. Defaults to 2.
            acti_type_1: activation type of the hidden squeeze layer. Defaults to ``("relu", {"inplace": True})``.
            acti_type_2: activation type of the output squeeze layer. Defaults to "sigmoid".

        Raises:
            ValueError: When ``r`` is nonpositive or larger than ``in_channels``.

        See also:

            :py:class:`monai.networks.layers.Act`

        """
        super(ChannelSELayer, self).__init__()

        self.add_residual = add_residual

        pool_type = Pool[Pool.ADAPTIVEAVG, spatial_dims]
        self.avg_pool = pool_type(1)  # spatial size (1, 1, ...)

        channels = int(in_channels // r)
        if channels <= 0:
            raise ValueError(f"r must be positive and smaller than in_channels, got r={r} in_channels={in_channels}.")

        act_1, act_1_args = split_args(acti_type_1)
        act_2, act_2_args = split_args(acti_type_2)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, channels, bias=True),
            Act[act_1](**act_1_args),
            nn.Linear(channels, in_channels, bias=True),
            Act[act_2](**act_2_args),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: in shape (batch, in_channels, spatial_1[, spatial_2, ...]).
        """
        b, c = x.shape[:2]
        y: torch.Tensor = self.avg_pool(x).view(b, c)
        y = self.fc(y).view([b, c] + [1] * (x.ndim - 2))
        result = x * y

        # Residual connection is moved here instead of providing an override of forward in ResidualSELayer since
        # Torchscript has an issue with using super().
        if self.add_residual:
            result += x

        return result


class ResidualSELayer(ChannelSELayer):
    """
    A "squeeze-and-excitation"-like layer with a residual connection::

        --+-- SE --o--
          |        |
          +--------+
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        r: int = 2,
        acti_type_1: Union[Tuple[str, Dict], str] = "leakyrelu",
        acti_type_2: Union[Tuple[str, Dict], str] = "relu",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
            in_channels: number of input channels.
            r: the reduction ratio r in the paper. Defaults to 2.
            acti_type_1: defaults to "leakyrelu".
            acti_type_2: defaults to "relu".

        See also:
            :py:class:`monai.networks.blocks.ChannelSELayer`
        """
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            r=r,
            acti_type_1=acti_type_1,
            acti_type_2=acti_type_2,
            add_residual=True,
        )


class SEBlock(nn.Module):
    """
    Residual module enhanced with Squeeze-and-Excitation::

        ----+- conv1 --  conv2 -- conv3 -- SE -o---
            |                                  |
            +---(channel project if needed)----+

    Re-implementation of the SE-Resnet block based on:
    "Hu et al., Squeeze-and-Excitation Networks, https://arxiv.org/abs/1709.01507".
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        n_chns_1: int,
        n_chns_2: int,
        n_chns_3: int,
        conv_param_1: Optional[Dict] = None,
        conv_param_2: Optional[Dict] = None,
        conv_param_3: Optional[Dict] = None,
        project: Optional[Convolution] = None,
        r: int = 2,
        acti_type_1: Union[Tuple[str, Dict], str] = ("relu", {"inplace": True}),
        acti_type_2: Union[Tuple[str, Dict], str] = "sigmoid",
        acti_type_final: Optional[Union[Tuple[str, Dict], str]] = ("relu", {"inplace": True}),
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
            in_channels: number of input channels.
            n_chns_1: number of output channels in the 1st convolution.
            n_chns_2: number of output channels in the 2nd convolution.
            n_chns_3: number of output channels in the 3rd convolution.
            conv_param_1: additional parameters to the 1st convolution.
                Defaults to ``{"kernel_size": 1, "norm": Norm.BATCH, "act": ("relu", {"inplace": True})}``
            conv_param_2: additional parameters to the 2nd convolution.
                Defaults to ``{"kernel_size": 3, "norm": Norm.BATCH, "act": ("relu", {"inplace": True})}``
            conv_param_3: additional parameters to the 3rd convolution.
                Defaults to ``{"kernel_size": 1, "norm": Norm.BATCH, "act": None}``
            project: in the case of residual chns and output chns doesn't match, a project
                (Conv) layer/block is used to adjust the number of chns. In SENET, it is
                consisted with a Conv layer as well as a Norm layer.
                Defaults to None (chns are matchable) or a Conv layer with kernel size 1.
            r: the reduction ratio r in the paper. Defaults to 2.
            acti_type_1: activation type of the hidden squeeze layer. Defaults to "relu".
            acti_type_2: activation type of the output squeeze layer. Defaults to "sigmoid".
            acti_type_final: activation type of the end of the block. Defaults to "relu".

        See also:

            :py:class:`monai.networks.blocks.ChannelSELayer`

        """
        super(SEBlock, self).__init__()

        if not conv_param_1:
            conv_param_1 = {"kernel_size": 1, "norm": Norm.BATCH, "act": ("relu", {"inplace": True})}
        self.conv1 = Convolution(
            dimensions=spatial_dims, in_channels=in_channels, out_channels=n_chns_1, **conv_param_1
        )

        if not conv_param_2:
            conv_param_2 = {"kernel_size": 3, "norm": Norm.BATCH, "act": ("relu", {"inplace": True})}
        self.conv2 = Convolution(dimensions=spatial_dims, in_channels=n_chns_1, out_channels=n_chns_2, **conv_param_2)

        if not conv_param_3:
            conv_param_3 = {"kernel_size": 1, "norm": Norm.BATCH, "act": None}
        self.conv3 = Convolution(dimensions=spatial_dims, in_channels=n_chns_2, out_channels=n_chns_3, **conv_param_3)

        self.se_layer = ChannelSELayer(
            spatial_dims=spatial_dims, in_channels=n_chns_3, r=r, acti_type_1=acti_type_1, acti_type_2=acti_type_2
        )

        if project is None and in_channels != n_chns_3:
            self.project = Conv[Conv.CONV, spatial_dims](in_channels, n_chns_3, kernel_size=1)
        elif project is None:
            self.project = nn.Identity()
        else:
            self.project = project

        if acti_type_final is not None:
            act_final, act_final_args = split_args(acti_type_final)
            self.act = Act[act_final](**act_final_args)
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: in shape (batch, in_channels, spatial_1[, spatial_2, ...]).
        """
        residual = self.project(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.se_layer(x)
        x += residual
        x = self.act(x)
        return x


class SEBottleneck(SEBlock):
    """
    Bottleneck for SENet154.
    """

    expansion = 4

    def __init__(
        self,
        spatial_dims: int,
        inplanes: int,
        planes: int,
        groups: int,
        reduction: int,
        stride: int = 1,
        downsample: Optional[Convolution] = None,
    ) -> None:

        conv_param_1 = {
            "strides": 1,
            "kernel_size": 1,
            "act": ("relu", {"inplace": True}),
            "norm": Norm.BATCH,
            "bias": False,
        }
        conv_param_2 = {
            "strides": stride,
            "kernel_size": 3,
            "act": ("relu", {"inplace": True}),
            "norm": Norm.BATCH,
            "groups": groups,
            "bias": False,
        }
        conv_param_3 = {"strides": 1, "kernel_size": 1, "act": None, "norm": Norm.BATCH, "bias": False}

        super(SEBottleneck, self).__init__(
            spatial_dims=spatial_dims,
            in_channels=inplanes,
            n_chns_1=planes * 2,
            n_chns_2=planes * 4,
            n_chns_3=planes * 4,
            conv_param_1=conv_param_1,
            conv_param_2=conv_param_2,
            conv_param_3=conv_param_3,
            project=downsample,
            r=reduction,
        )


class SEResNetBottleneck(SEBlock):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `strides=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """

    expansion = 4

    def __init__(
        self,
        spatial_dims: int,
        inplanes: int,
        planes: int,
        groups: int,
        reduction: int,
        stride: int = 1,
        downsample: Optional[Convolution] = None,
    ) -> None:

        conv_param_1 = {
            "strides": stride,
            "kernel_size": 1,
            "act": ("relu", {"inplace": True}),
            "norm": Norm.BATCH,
            "bias": False,
        }
        conv_param_2 = {
            "strides": 1,
            "kernel_size": 3,
            "act": ("relu", {"inplace": True}),
            "norm": Norm.BATCH,
            "groups": groups,
            "bias": False,
        }
        conv_param_3 = {"strides": 1, "kernel_size": 1, "act": None, "norm": Norm.BATCH, "bias": False}

        super(SEResNetBottleneck, self).__init__(
            spatial_dims=spatial_dims,
            in_channels=inplanes,
            n_chns_1=planes,
            n_chns_2=planes,
            n_chns_3=planes * 4,
            conv_param_1=conv_param_1,
            conv_param_2=conv_param_2,
            conv_param_3=conv_param_3,
            project=downsample,
            r=reduction,
        )


class SEResNeXtBottleneck(SEBlock):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """

    expansion = 4

    def __init__(
        self,
        spatial_dims: int,
        inplanes: int,
        planes: int,
        groups: int,
        reduction: int,
        stride: int = 1,
        downsample: Optional[Convolution] = None,
        base_width: int = 4,
    ) -> None:

        conv_param_1 = {
            "strides": 1,
            "kernel_size": 1,
            "act": ("relu", {"inplace": True}),
            "norm": Norm.BATCH,
            "bias": False,
        }
        conv_param_2 = {
            "strides": stride,
            "kernel_size": 3,
            "act": ("relu", {"inplace": True}),
            "norm": Norm.BATCH,
            "groups": groups,
            "bias": False,
        }
        conv_param_3 = {"strides": 1, "kernel_size": 1, "act": None, "norm": Norm.BATCH, "bias": False}
        width = math.floor(planes * (base_width / 64)) * groups

        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=inplanes,
            n_chns_1=width,
            n_chns_2=width,
            n_chns_3=planes * 4,
            conv_param_1=conv_param_1,
            conv_param_2=conv_param_2,
            conv_param_3=conv_param_3,
            project=downsample,
            r=reduction,
        )
