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

from monai.networks.blocks import Convolution
from monai.networks.layers import Act
from monai.networks.utils import normal_init


class MultiScalePatchDiscriminator(nn.Sequential):
    """
    Multi-scale Patch-GAN discriminator based on Pix2PixHD:
    High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs (https://arxiv.org/abs/1711.11585)

    The Multi-scale discriminator made up of several PatchGAN discriminators, that process the images
    at different spatial scales.

    Args:
        num_d: number of discriminators
        num_layers_d: number of Convolution layers (Conv + activation + normalisation + [dropout]) in the first
            discriminator. Each subsequent discriminator has one additional layer, meaning the output size is halved.
        spatial_dims: number of spatial dimensions (1D, 2D etc.)
        channels: number of filters in the first convolutional layer (doubled for each subsequent layer)
        in_channels: number of input channels
        out_channels: number of output channels in each discriminator
        kernel_size: kernel size of the convolution layers
        activation: activation layer type
        norm: normalisation type
        bias: introduction of layer bias
        dropout: probability of dropout applied, defaults to 0.
        minimum_size_im: minimum spatial size of the input image. Introduced to make sure the architecture
            requested isn't going to downsample the input image beyond value of 1.
        last_conv_kernel_size: kernel size of the last convolutional layer.
    """

    def __init__(
        self,
        num_d: int,
        num_layers_d: int,
        spatial_dims: int,
        channels: int,
        in_channels: int,
        out_channels: int = 1,
        kernel_size: int = 4,
        activation: str | tuple = (Act.LEAKYRELU, {"negative_slope": 0.2}),
        norm: str | tuple = "BATCH",
        bias: bool = False,
        dropout: float | tuple = 0.0,
        minimum_size_im: int = 256,
        last_conv_kernel_size: int = 1,
    ) -> None:
        super().__init__()
        self.num_d = num_d
        self.num_layers_d = num_layers_d
        self.num_channels = channels
        self.padding = tuple([int((kernel_size - 1) / 2)] * spatial_dims)
        for i_ in range(self.num_d):
            num_layers_d_i = self.num_layers_d * (i_ + 1)
            output_size = float(minimum_size_im) / (2**num_layers_d_i)
            if output_size < 1:
                raise AssertionError(
                    f"Your image size is too small to take in up to {i_} discriminators with num_layers = {num_layers_d_i}."
                    "Please reduce num_layers, reduce num_D or enter bigger images."
                )
            subnet_d = PatchDiscriminator(
                spatial_dims=spatial_dims,
                channels=self.num_channels,
                in_channels=in_channels,
                out_channels=out_channels,
                num_layers_d=num_layers_d_i,
                kernel_size=kernel_size,
                activation=activation,
                norm=norm,
                bias=bias,
                padding=self.padding,
                dropout=dropout,
                last_conv_kernel_size=last_conv_kernel_size,
            )

            self.add_module("discriminator_%d" % i_, subnet_d)

    def forward(self, i: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        """
        Args:
            i: Input tensor

        Returns:
            list of outputs and another list of lists with the intermediate features
            of each discriminator.
        """

        out: list[torch.Tensor] = []
        intermediate_features: list[list[torch.Tensor]] = []
        for disc in self.children():
            out_d: list[torch.Tensor] = disc(i)
            out.append(out_d[-1])
            intermediate_features.append(out_d[:-1])

        return out, intermediate_features


class PatchDiscriminator(nn.Sequential):
    """
    Patch-GAN discriminator based on Pix2PixHD:
    High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs (https://arxiv.org/abs/1711.11585)


    Args:
        spatial_dims: number of spatial dimensions (1D, 2D etc.)
        channels: number of filters in the first convolutional layer (doubled for each subsequent layer)
        in_channels: number of input channels
        out_channels: number of output channels
        num_layers_d: number of Convolution layers (Conv + activation + normalisation + [dropout]) in the discriminator.
        kernel_size: kernel size of the convolution layers
        act: activation type and arguments. Defaults to LeakyReLU.
        norm: feature normalization type and arguments. Defaults to batch norm.
        bias: whether to have a bias term in convolution blocks. Defaults to False.
        padding: padding to be applied to the convolutional layers
        dropout: proportion of dropout applied, defaults to 0.
        last_conv_kernel_size: kernel size of the last convolutional layer.
    """

    def __init__(
        self,
        spatial_dims: int,
        channels: int,
        in_channels: int,
        out_channels: int = 1,
        num_layers_d: int = 3,
        kernel_size: int = 4,
        activation: str | tuple = (Act.LEAKYRELU, {"negative_slope": 0.2}),
        norm: str | tuple = "BATCH",
        bias: bool = False,
        padding: int | Sequence[int] = 1,
        dropout: float | tuple = 0.0,
        last_conv_kernel_size: int | None = None,
    ) -> None:
        super().__init__()
        self.num_layers_d = num_layers_d
        self.num_channels = channels
        if last_conv_kernel_size is None:
            last_conv_kernel_size = kernel_size

        self.add_module(
            "initial_conv",
            Convolution(
                spatial_dims=spatial_dims,
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=channels,
                act=activation,
                bias=True,
                norm=None,
                dropout=dropout,
                padding=padding,
                strides=2,
            ),
        )

        input_channels = channels
        output_channels = channels * 2

        # Initial Layer
        for l_ in range(self.num_layers_d):
            if l_ == self.num_layers_d - 1:
                stride = 1
            else:
                stride = 2
            layer = Convolution(
                spatial_dims=spatial_dims,
                kernel_size=kernel_size,
                in_channels=input_channels,
                out_channels=output_channels,
                act=activation,
                bias=bias,
                norm=norm,
                dropout=dropout,
                padding=padding,
                strides=stride,
            )
            self.add_module("%d" % l_, layer)
            input_channels = output_channels
            output_channels = output_channels * 2

        # Final layer
        self.add_module(
            "final_conv",
            Convolution(
                spatial_dims=spatial_dims,
                kernel_size=last_conv_kernel_size,
                in_channels=input_channels,
                out_channels=out_channels,
                bias=True,
                conv_only=True,
                padding=int((last_conv_kernel_size - 1) / 2),
                dropout=0.0,
                strides=1,
            ),
        )

        self.apply(normal_init)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: input tensor

        Returns:
            list of intermediate features, with the last element being the output.
        """
        out = [x]
        for submodel in self.children():
            intermediate_output = submodel(out[-1])
            out.append(intermediate_output)

        return out[1:]
