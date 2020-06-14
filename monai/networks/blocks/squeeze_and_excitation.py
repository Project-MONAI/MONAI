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

import torch.nn as nn

from monai.networks.layers.factories import Pool

SUPPORTED_ACTI_1 = {"relu": nn.ReLU, "relu6": nn.ReLU6, "leakyrelu": nn.LeakyReLU}

SUPPORTED_ACTI_2 = {
    "relu": nn.ReLU,
    "prelu": nn.PReLU,
    "relu6": nn.ReLU6,
    "leakyrelu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
}


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
            acti_type_1: activation type of the hidden squeeze layer.
                Supported types: "relu", "relu6", "leakyrelu".
                Defaults to "relu".
            acti_type_2: activation type of the output squeeze layer.
                Supported types: "relu", "prelu", "leakyrelu", "sigmoid".
                Defaults to "sigmoid".
        """
        super(ChannelSELayer, self).__init__()

        pool_type = Pool[Pool.ADAPTIVEAVG, spatial_dims]
        self.avg_pool = pool_type(1)  # spatial size (1, 1, ...)

        channels = int(in_channels // r)
        if channels <= 0:
            raise ValueError("r must be a positive number smaller than `in_channels`.")
        self.fc = nn.Sequential(
            nn.Linear(in_channels, channels, bias=True),
            SUPPORTED_ACTI_1[acti_type_1](inplace=True),
            nn.Linear(channels, in_channels, bias=True),
            SUPPORTED_ACTI_2[acti_type_2](),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view([b, c] + [1] * (x.ndim - 2))
        return x * y


class ResidualSELayer(nn.Module):
    """
    A "squeeze-and-excitation"-like layer with a residual connection.

    See also ::py:class:`monai.networks.blocks.ChannelSELayer`.
    """

    def __init__(self, spatial_dims: int, in_channels: int, r: int = 2):
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
            in_channels: number of input channels.
            r: the reduction ratio r in the paper. Defaults to 2.
        """
        super(ResidualSELayer, self).__init__()
        self.channel_se_layer = ChannelSELayer(
            spatial_dims=spatial_dims, in_channels=in_channels, r=r, acti_type_1="leakyrelu", acti_type_2="relu"
        )

    def forward(self, x):
        return x + self.channel_se_layer(x)
