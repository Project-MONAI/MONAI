# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

from monai.networks.layers.factories import Act, Pool


class ChannelSELayer(nn.Module):
    """
    Re-implementation of the Squeeze-and-Excitation block based on:
    "Hu et al., Squeeze-and-Excitation Networks, https://arxiv.org/abs/1709.01507".
    """

    def __init__(
        self, spatial_dims: int, in_channels: int, r: int = 2, acti_type_1: str = "relu", acti_type_2: str = "sigmoid"
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
            in_channels: number of input channels.
            r: the reduction ratio r in the paper. Defaults to 2.
            acti_type_1: activation type of the hidden squeeze layer. Defaults to "relu".
            acti_type_2: activation type of the output squeeze layer. Defaults to "sigmoid".

        Raises:
            ValueError: r must be a positive number smaller than `in_channels`.

        """
        super(ChannelSELayer, self).__init__()

        pool_type = Pool[Pool.ADAPTIVEAVG, spatial_dims]
        self.avg_pool = pool_type(1)  # spatial size (1, 1, ...)

        channels = int(in_channels // r)
        if channels <= 0:
            raise ValueError("r must be a positive number smaller than `in_channels`.")
        self.fc = nn.Sequential(
            nn.Linear(in_channels, channels, bias=True),
            Act[acti_type_1](inplace=True),
            nn.Linear(channels, in_channels, bias=True),
            Act[acti_type_2](),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: in shape (batch, channel, spatial_1[, spatial_2, ...]).
        """
        b, c = x.shape[:2]
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view([b, c] + [1] * (x.ndim - 2))
        return x * y


class ResidualSELayer(ChannelSELayer):
    """
    A "squeeze-and-excitation"-like layer with a residual connection.
    """

    def __init__(
        self, spatial_dims: int, in_channels: int, r: int = 2, acti_type_1: str = "leakyrelu", acti_type_2: str = "relu"
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
            in_channels: number of input channels.
            r: the reduction ratio r in the paper. Defaults to 2.
            acti_type_1: defaults to "leakyrelu".
            acti_type_2: defaults to "relu".

        See also ::py:class:`monai.networks.blocks.ChannelSELayer`.
        """
        super().__init__(
            spatial_dims=spatial_dims, in_channels=in_channels, r=r, acti_type_1=acti_type_1, acti_type_2=acti_type_2
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: in shape (batch, channel, spatial_1[, spatial_2, ...]).
        """
        return x + super().forward(x)
