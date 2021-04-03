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

from typing import List, Optional, Sequence, Tuple, Type, Union

import torch
from torch import nn
from torch.nn import functional as F

from monai.networks.blocks import Convolution
from monai.networks.layers import Conv, Norm, Pool, same_padding


def get_conv_block(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Sequence[int], int] = 3,
    strides: int = 1,
    padding: Optional[Union[Tuple[int, ...], int]] = None,
    act: Optional[Union[Tuple, str]] = "RELU",
    norm: Optional[Union[Tuple, str]] = "BATCH",
    initializer: Optional[str] = "kaiming_uniform",
) -> nn.Module:
    if padding is None:
        padding = same_padding(kernel_size)
    conv_block = Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        strides=strides,
        act=act,
        norm=norm,
        bias=False,
        conv_only=False,
        padding=padding,
    )
    conv_type: Type[Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]] = Conv[Conv.CONV, spatial_dims]
    for m in conv_block.modules():
        if isinstance(m, conv_type):
            if initializer == "kaiming_uniform":
                nn.init.kaiming_normal_(torch.as_tensor(m.weight))
            elif initializer == "zeros":
                nn.init.zeros_(torch.as_tensor(m.weight))
            else:
                raise ValueError(
                    f"initializer {initializer} is not supported, " "currently supporting kaiming_uniform and zeros"
                )
    return conv_block


def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Sequence[int], int] = 3,
) -> nn.Module:
    padding = same_padding(kernel_size)
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        bias=False,
        conv_only=True,
        padding=padding,
    )


class RegistrationResidualConvBlock(nn.Module):
    """
    A block with skip links and layer - norm - activation.
    Only changes the number of channels, the spatial size is kept same.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        kernel_size: int = 3,
    ):
        """

        Args:
            spatial_dims: number of spatial dimensions
            in_channels: number of input channels
            out_channels: number of output channels
            num_layers: number of layers inside the block
            kernel_size: kernel_size
        """
        super(RegistrationResidualConvBlock, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                get_conv_layer(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                )
                for i in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([Norm[Norm.BATCH, spatial_dims](out_channels) for _ in range(num_layers)])
        self.acts = nn.ModuleList([nn.ReLU() for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: Tensor in shape (batch, ``in_channels``, insize_1, insize_2, [insize_3])

        Returns:
            Tensor in shape (batch, ``out_channels``, insize_1, insize_2, [insize_3]),
            with the same spatial size as ``x``
        """
        skip = x
        for i, (conv, norm, act) in enumerate(zip(self.layers, self.norms, self.acts)):
            x = conv(x)
            x = norm(x)
            if i == self.num_layers - 1:
                # last block
                x = x + skip
            x = act(x)
        return x


class RegistrationDownSampleBlock(nn.Module):
    """
    A down-sample module used in RegUNet to half the spatial size.
    The number of channels is kept same.

    Adapted from:
        DeepReg (https://github.com/DeepRegNet/DeepReg)
    """

    def __init__(
        self,
        spatial_dims: int,
        channels: int,
        pooling: bool,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            channels: channels
            pooling: use MaxPool if True, strided conv if False
        """
        super(RegistrationDownSampleBlock, self).__init__()
        if pooling:
            self.layer = Pool[Pool.MAX, spatial_dims](kernel_size=2)
        else:
            self.layer = get_conv_block(
                spatial_dims=spatial_dims,
                in_channels=channels,
                out_channels=channels,
                kernel_size=2,
                strides=2,
                padding=0,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Halves the spatial dimensions and keeps the same channel.
        output in shape (batch, ``channels``, insize_1 / 2, insize_2 / 2, [insize_3 / 2]),

        Args:
            x: Tensor in shape (batch, ``channels``, insize_1, insize_2, [insize_3])

        Raises:
            ValueError: when input spatial dimensions are not even.
        """
        for i in x.shape[2:]:
            if i % 2 != 0:
                raise ValueError("expecting x spatial dimensions be even, " f"got x of shape {x.shape}")
        out: torch.Tensor = self.layer(x)
        return out


def get_deconv_block(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
) -> nn.Module:
    return Convolution(
        dimensions=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=2,
        act="RELU",
        norm="BATCH",
        bias=False,
        is_transposed=True,
        padding=1,
        output_padding=1,
    )


class RegistrationExtractionBlock(nn.Module):
    """
    The Extraction Block used in RegUNet.
    Extracts feature from each ``extract_levels`` and takes the average.
    """

    def __init__(
        self,
        spatial_dims: int,
        extract_levels: Tuple[int],
        num_channels: Union[Tuple[int], List[int]],
        out_channels: int,
        kernel_initializer: Optional[str] = "kaiming_uniform",
        activation: Optional[str] = None,
    ):
        """

        Args:
            spatial_dims: number of spatial dimensions
            extract_levels: spatial levels to extract feature from, 0 refers to the input scale
            num_channels: number of channels at each scale level,
                List or Tuple of length equals to `depth` of the RegNet
            out_channels: number of output channels
            kernel_initializer: kernel initializer
            activation: kernel activation function
        """
        super(RegistrationExtractionBlock, self).__init__()
        self.extract_levels = extract_levels
        self.max_level = max(extract_levels)
        self.layers = nn.ModuleList(
            [
                get_conv_block(
                    spatial_dims=spatial_dims,
                    in_channels=num_channels[d],
                    out_channels=out_channels,
                    norm=None,
                    act=activation,
                    initializer=kernel_initializer,
                )
                for d in extract_levels
            ]
        )

    def forward(self, x: List[torch.Tensor], image_size: List[int]) -> torch.Tensor:
        """

        Args:
            x: Decoded feature at different spatial levels, sorted from deep to shallow
            image_size: output image size

        Returns:
            Tensor of shape (batch, `out_channels`, size1, size2, size3), where (size1, size2, size3) = ``image_size``
        """
        feature_list = [
            F.interpolate(
                layer(x[self.max_level - level]),
                size=image_size,
            )
            for layer, level in zip(self.layers, self.extract_levels)
        ]
        out: torch.Tensor = torch.mean(torch.stack(feature_list, dim=0), dim=0)
        return out
