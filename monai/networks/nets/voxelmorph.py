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
import torch.nn.functional as F

from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.upsample import UpSample
from monai.networks.blocks.warp import DVF2DDF, Warp
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import alias, export

__all__ = ["VoxelMorph", "voxelmorph"]


@export("monai.networks.nets")
@alias("voxelmorph")
class VoxelMorph(nn.Module):
    """
    VoxelMorph network for medical image registration as described in https://arxiv.org/pdf/1809.05231.pdf.
    For more details, please refer to VoxelMorph: A Learning Framework for Deformable Medical Image Registration
    Guha Balakrishnan, Amy Zhao, Mert R. Sabuncu, John Guttag, Adrian V. Dalca
    IEEE TMI: Transactions on Medical Imaging. 2019. eprint arXiv:1809.05231.

    A pair of images (moving and fixed) are concatenated along the channel dimension and passed through
    a UNet. The output of the UNet is then passed through a series of convolution blocks to produce the final prediction
    of the displacement field (DDF) in the non-diffeomorphic variant (i.e. when `int_steps` is set to 0) or the
    stationary velocity field (DVF) in the diffeomorphic variant (i.e. when `int_steps` is set to a positive integer).
    The DVF is then converted to a DDF using the `DVF2DDF` module. Finally, the DDF is used to warp the moving image
    to the fixed image using the `Warp` module. Optionally, the integration from DVF to DDF can be performed on reduced
    resolution by specifying `half_res` to be True, in which case the output DVF from the UNet is first linearly
    interpolated to half resolution before being passed to the `DVF2DDF` module. The output DDF is then linearly
    interpolated again back to full resolution before being used in the `Warp` module.

    In the original implementation, downsample is achieved through maxpooling, here one has the option to use either
    maxpooling or strided convolution for downsampling. The default is to use maxpooling as it is consistent with the
    original implementation. Note that for upsampling, the authors of VoxelMorph used nearest neighbor interpolation
    instead of transposed convolution. In this implementation, only nearest neighbor interpolation is supported in order
    to be consistent with the original implementation.

    Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of channels in the input volume after concatenation of moving and fixed images.
            unet_out_channels: number of channels in the output of the UNet.
            channels: number of channels in each layer of the UNet. See the following example for more details.
            final_conv_channels: number of channels in each layer of the final convolution block.
            final_conv_act: activation type for the final convolution block. Defaults to LeakyReLU.
                    Since VoxelMorph was originally implemented in tensorflow where the default negative slope for LeakyReLU
                    was 0.2, we use the same default value here.
            int_steps: number of integration steps. Defaults to 7. If set to 0, the network will be non-diffeomorphic.
            kernel_size: kernel size for all convolution layers in the UNet. Defaults to 3.
            up_kernel_size: kernel size for all convolution layers in the upsampling path of the UNet. Defaults to 3.
            act: activation type for all convolution layers in the UNet. Defaults to LeakyReLU with negative slope 0.2.
            norm: feature normalization type and arguments for all convolution layers in the UNet. Defaults to None.
            dropout: dropout ratio for all convolution layers in the UNet. Defaults to 0.0 (no dropout).
            bias: whether to use bias in all convolution layers in the UNet. Defaults to True.
            half_res: whether to perform integration on half resolution. Defaults to False.
            use_maxpool: whether to use maxpooling in the downsampling path of the UNet. Defaults to True.
                    Using maxpooling is the consistent with the original implementation of VoxelMorph.
                    But one can optionally use strided convolution instead (i.e. set `use_maxpool` to False).
            adn_ordering: ordering of activation, dropout, and normalization. Defaults to "NDA".

    Example::

            from monai.networks.nets import VoxelMorph

            # VoxelMorph network as it is in the original paper https://arxiv.org/pdf/1809.05231.pdf
            net = VoxelMorph(
                    spatial_dims=3,
                    in_channels=2,
                    unet_out_channels=32,
                    channels=(16, 32, 32, 32, 32, 32),  # this indicates the down block at the top takes 16 channels as
                                                        # input, the corresponding up block at the top produces 32
                                                        # channels as output, the second down block takes 32 channels as
                                                        # input, and the corresponding up block at the same level
                                                        # produces 32 channels as output, etc.
                    final_conv_channels=(16, 16)
            )

            # A forward pass through the network would look something like this
            moving = torch.randn(1, 1, 160, 192, 224)
            fixed = torch.randn(1, 1, 160, 192, 224)
            warped, ddf = net(moving, fixed)
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        unet_out_channels: int,
        channels: Sequence[int],
        final_conv_channels: Sequence[int],
        final_conv_act: tuple | str | None = "LEAKYRELU",
        int_steps: int = 7,
        kernel_size: Sequence[int] | int = 3,
        up_kernel_size: Sequence[int] | int = 3,
        act: tuple | str = "LEAKYRELU",
        norm: tuple | str | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        half_res: bool = False,
        use_maxpool: bool = True,
        adn_ordering: str = "NDA",
    ) -> None:
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("spatial_dims must be either 2 or 3.")
        if in_channels % 2 != 0:
            raise ValueError("in_channels must be divisible by 2.")
        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        if len(channels) % 2 != 0:
            raise ValueError("the elements of `channels` should be specified in pairs.")
        if isinstance(kernel_size, Sequence) and len(kernel_size) != spatial_dims:
            raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence) and len(up_kernel_size) != spatial_dims:
            raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        # UNet args
        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.channels = channels
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.act = (
            ("leakyrelu", {"negative_slope": 0.2, "inplace": True})
            if isinstance(act, str) and act.upper() == "LEAKYRELU"
            else act
        )
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        # VoxelMorph specific args
        self.unet_out_channels = unet_out_channels
        self.half_res = half_res
        self.use_maxpool = use_maxpool

        # final convolutions args
        self.final_conv_channels = final_conv_channels
        self.final_conv_act = (
            ("leakyrelu", {"negative_slope": 0.2, "inplace": True})
            if isinstance(final_conv_act, str) and final_conv_act.upper() == "LEAKYRELU"
            else final_conv_act
        )

        # integration args
        self.int_steps = int_steps
        self.diffeomorphic = True if self.int_steps > 0 else False

        def _create_block(inc: int, outc: int, channels: Sequence[int], is_top: bool) -> nn.Module:
            """
            Builds the UNet structure recursively.

            Args:
                    inc: number of input channels.
                    outc: number of output channels.
                    channels: sequence of channels for each pair of down and up layers.
                    is_top: True if this is the top block.
            """

            next_c_in, next_c_out = channels[0:2]
            upc = next_c_in + next_c_out

            subblock: nn.Module

            if len(channels) > 2:
                subblock = _create_block(next_c_in, next_c_out, channels[2:], is_top=False)  # continue recursion down
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(next_c_in, next_c_out)

            down = self._get_down_layer(inc, next_c_in, is_top)  # create layer in downsampling path
            up = self._get_up_layer(upc, outc, is_top)  # create layer in upsampling path

            return self._get_connection_block(down, up, subblock)

        def _create_final_conv(inc: int, outc: int, channels: Sequence[int]) -> nn.Module:
            """
            Builds the final convolution blocks.

            Args:
                    inc: number of input channels, should be the same as `unet_out_channels`.
                    outc: number of output channels, should be the same as `spatial_dims`.
                    channels: sequence of channels for each convolution layer.

            Note: there is no activation after the last convolution layer as per the original implementation.
            """

            mod: nn.Module = nn.Sequential()

            for i, c in enumerate(channels):
                mod.add_module(
                    f"final_conv_{i}",
                    Convolution(
                        self.dimensions,
                        inc,
                        c,
                        kernel_size=self.kernel_size,
                        act=self.final_conv_act,
                        norm=self.norm,
                        dropout=self.dropout,
                        bias=self.bias,
                        adn_ordering=self.adn_ordering,
                    ),
                )
                inc = c

            mod.add_module(
                "final_conv_out",
                Convolution(
                    self.dimensions,
                    inc,
                    outc,
                    kernel_size=self.kernel_size,
                    act=None,
                    norm=self.norm,
                    dropout=self.dropout,
                    bias=self.bias,
                    adn_ordering=self.adn_ordering,
                ),
            )

            return mod

        self.net = nn.Sequential(
            _create_block(in_channels, unet_out_channels, self.channels, is_top=True),
            _create_final_conv(unet_out_channels, self.dimensions, self.final_conv_channels),
        )

        # create helpers
        if self.diffeomorphic:
            self.dvf2ddf = DVF2DDF(num_steps=self.int_steps, mode="bilinear", padding_mode="zeros")
        self.warp = Warp(mode="bilinear", padding_mode="zeros")

    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        """
        Returns the block object defining a layer of the UNet structure including the implementation of the skip
        between encoding (down) and decoding (up) sides of the network.

        Args:
                down_path: encoding half of the layer
                up_path: decoding half of the layer
                subblock: block defining the next layer in the network.
        Returns: block for this layer: `nn.Sequential(down_path, SkipConnection(subblock), up_path)`
        """

        return nn.Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_down_layer(self, in_channels: int, out_channels: int, is_top: bool) -> nn.Module:
        """
        In each down layer, the input is first downsampled using maxpooling,
        then passed through a convolution block, unless this is the top layer
        in which case the input is passed through a convolution block only
        without maxpooling first.

        Args:
                in_channels: number of input channels.
                out_channels: number of output channels.
                is_top: True if this is the top block.
        """

        mod: Convolution | nn.Sequential

        strides = 1 if self.use_maxpool or is_top else 2

        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )

        if self.use_maxpool and not is_top:
            mod = (
                nn.Sequential(nn.MaxPool3d(kernel_size=2, stride=2), mod)
                if self.dimensions == 3
                else nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), mod)
            )

        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Bottom layer (bottleneck) in voxelmorph consists of a typical down layer followed by an upsample layer.

        Args:
                in_channels: number of input channels.
                out_channels: number of output channels.
        """

        mod: nn.Module
        upsample: nn.Module

        mod = self._get_down_layer(in_channels, out_channels, is_top=False)

        upsample = UpSample(
            self.dimensions,
            out_channels,
            out_channels,
            scale_factor=2,
            mode="nontrainable",
            interp_mode="nearest",
            align_corners=None,  # required to use with interp_mode="nearest"
        )

        return nn.Sequential(mod, upsample)

    def _get_up_layer(self, in_channels: int, out_channels: int, is_top: bool) -> nn.Module:
        """
        In each up layer, the input is passed through a convolution block before upsampled,
        unless this is the top layer in which case the input is passed through a convolution block only
        without upsampling.

        Args:
                in_channels: number of input channels.
                out_channels: number of output channels.
                is_top: True if this is the top block.
        """

        mod: Convolution | nn.Sequential

        strides = 1

        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            # conv_only=is_top,
            is_transposed=False,
            adn_ordering=self.adn_ordering,
        )

        if not is_top:
            mod = nn.Sequential(
                mod,
                UpSample(
                    self.dimensions,
                    out_channels,
                    out_channels,
                    scale_factor=2,
                    mode="nontrainable",
                    interp_mode="nearest",
                    align_corners=None,  # required to use with interp_mode="nearest"
                ),
            )

        return mod

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.net(torch.cat([moving, fixed], dim=1))

        if self.half_res:
            x = F.interpolate(x, scale_factor=0.5, mode="trilinear", align_corners=True) * 2.0

        if self.diffeomorphic:
            x = self.dvf2ddf(x)

        if self.half_res:
            x = F.interpolate(x * 0.5, scale_factor=2.0, mode="trilinear", align_corners=True)

        return self.warp(moving, x), x


voxelmorph = VoxelMorph
