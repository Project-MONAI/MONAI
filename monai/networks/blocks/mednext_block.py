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

all = ["MedNeXtBlock", "MedNeXtDownBlock", "MedNeXtUpBlock", "MedNeXtOutBlock"]


class MedNeXtBlock(nn.Module):
    """
    MedNeXtBlock class for the MedNeXt model.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        expansion_ratio (int): Expansion ratio for the block. Defaults to 4.
        kernel_size (int): Kernel size for convolutions. Defaults to 7.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int,
        kernel_size: int,
    ):

        super().__init__()

        # First convolution layer with DepthWise Convolutions
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels,
        )

        # Normalization Layer.
        self.norm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels)  # type: ignore

        # Second convolution (Expansion) layer with Conv3D 1x1x1
        self.conv2 = nn.Conv2d(
            in_channels=in_channels, out_channels=expansion_ratio * in_channels, kernel_size=1, stride=1, padding=0
        )

        # GeLU activations
        self.act = nn.GELU()

        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = nn.Conv2d(
            in_channels=expansion_ratio * in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        """
        Forward pass of the MedNeXtBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))

        x1 = self.conv3(x1)
        x1 = x + x1

        return x1


class MedNeXtDownBlock(MedNeXtBlock):
    """
    MedNeXtDownBlock class for downsampling in the MedNeXt model.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        expansion_ratio (int): Expansion ratio for the block. Defaults to 4.
        kernel_size (int): Kernel size for convolutions. Defaults to 7.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int = 4,
        kernel_size: int = 7,
    ):

        super().__init__(
            in_channels,
            out_channels,
            expansion_ratio,
            kernel_size,
        )

        self.res_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=2,
        )

        # Overwrite the first convolution layer with stride 2
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x):
        """
        Forward pass of the MedNeXtDownBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))

        x1 = self.conv3(x1)
        x1 = x + x1

        res = self.res_conv(x)
        x1 = x1 + res

        return x1


class MedNeXtUpBlock(MedNeXtBlock):
    """
    MedNeXtUpBlock class for upsampling in the MedNeXt model.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        expansion_ratio (int): Expansion ratio for the block. Defaults to 4.
        kernel_size (int): Kernel size for convolutions. Defaults to 7.
        use_residual_connection (bool): Whether to use residual connection. Defaults to False.
        norm_type (str): Type of normalization to use. Defaults to "group".
        dim (str): Dimension of the input. Can be "2d" or "3d". Defaults to "3d".
        global_resp_norm (bool): Whether to use global response normalization. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int = 4,
        kernel_size: int = 7,
    ):
        super().__init__(
            in_channels,
            out_channels,
            expansion_ratio,
            kernel_size,
        )

        self.res_conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=2,
        )

        self.conv1 = nn.ConvTranspose2d(  # type: ignore
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x):
        """
        Forward pass of the MedNeXtUpBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))

        x1 = self.conv3(x1)
        x1 = x + x1

        x1 = torch.nn.functional.pad(x1, (1, 0, 1, 0))

        res = self.res_conv(x)
        res = torch.nn.functional.pad(res, (1, 0, 1, 0))

        x1 = x1 + res

        return x1


class MedNeXtOutBlock(nn.Module):
    """
    MedNeXtOutBlock class for the output block in the MedNeXt model.

    Args:
        in_channels (int): Number of input channels.
        n_classes (int): Number of output classes.
        dim (str): Dimension of the input. Can be "2d" or "3d".
    """

    def __init__(self, in_channels, n_classes):
        super().__init__()

        self.conv_out = nn.ConvTranspose2d(in_channels, n_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the MedNeXtOutBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.conv_out(x)
