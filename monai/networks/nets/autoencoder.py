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

from typing import Any, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm

__all__ = ["AutoEncoder"]


class AutoEncoder(nn.Module):
    """
    Simple definition of an autoencoder and base class for the architecture implementing
    :py:class:`monai.networks.nets.VarAutoEncoder`. The network is composed of an encode sequence of blocks, followed
    by an intermediary sequence of blocks, and finally a decode sequence of blocks. The encode and decode blocks are
    default :py:class:`monai.networks.blocks.Convolution` instances with the encode blocks having the given stride
    and the decode blocks having transpose convolutions with the same stride. If `num_res_units` is given residual
    blocks are used instead.

    By default the intermediary sequence is empty but if `inter_channels` is given to specify the output channels of
    blocks then this will be become a sequence of Convolution blocks or of residual blocks if `num_inter_units` is
    given. The optional parameter `inter_dilations` can be used to specify the dilation values of the convolutions in
    these blocks, this allows a network to use dilated kernels in this  middle section. Since the intermediary section
    isn't meant to change the size of the output the strides for all these kernels is 1.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        channels: sequence of channels. Top block first. The length of `channels` should be no less than 2.
        strides: sequence of convolution strides. The length of `stride` should equal to `len(channels) - 1`.
        kernel_size: convolution kernel size, the value(s) should be odd. If sequence,
            its length should equal to dimensions. Defaults to 3.
        up_kernel_size: upsampling convolution kernel size, the value(s) should be odd. If sequence,
            its length should equal to dimensions. Defaults to 3.
        num_res_units: number of residual units. Defaults to 0.
        inter_channels: sequence of channels defining the blocks in the intermediate layer between encode and decode.
        inter_dilations: defines the dilation value for each block of the intermediate layer. Defaults to 1.
        num_inter_units: number of residual units for each block of the intermediate layer. Defaults to 0.
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        bias: whether to have a bias term in convolution blocks. Defaults to True.
            According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
            if a conv layer is directly followed by a batch norm layer, bias should be False.

    Examples::

        from monai.networks.nets import AutoEncoder

        # 3 layers each down/up sampling their inputs by a factor 2 with no intermediate layer
        net = AutoEncoder(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(2, 4, 8),
            strides=(2, 2, 2)
        )

        # 1 layer downsampling by 2, followed by a sequence of residual units with 2 convolutions defined by
        # progressively increasing dilations, then final upsample layer
        net = AutoEncoder(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=(4,),
                strides=(2,),
                inter_channels=(8, 8, 8),
                inter_dilations=(1, 2, 4),
                num_inter_units=2
            )

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        inter_channels: Optional[list] = None,
        inter_dilations: Optional[list] = None,
        num_inter_units: int = 2,
        act: Optional[Union[Tuple, str]] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: Optional[Union[Tuple, str, float]] = None,
        bias: bool = True,
    ) -> None:

        super().__init__()
        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = list(channels)
        self.strides = list(strides)
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.num_inter_units = num_inter_units
        self.inter_channels = inter_channels if inter_channels is not None else []
        self.inter_dilations = list(inter_dilations or [1] * len(self.inter_channels))

        # The number of channels and strides should match
        if len(channels) != len(strides):
            raise ValueError("Autoencoder expects matching number of channels and strides")

        self.encoded_channels = in_channels
        decode_channel_list = list(channels[-2::-1]) + [out_channels]

        self.encode, self.encoded_channels = self._get_encode_module(self.encoded_channels, channels, strides)
        self.intermediate, self.encoded_channels = self._get_intermediate_module(self.encoded_channels, num_inter_units)
        self.decode, _ = self._get_decode_module(self.encoded_channels, decode_channel_list, strides[::-1] or [1])

    def _get_encode_module(
        self, in_channels: int, channels: Sequence[int], strides: Sequence[int]
    ) -> Tuple[nn.Sequential, int]:
        """
        Returns the encode part of the network by building up a sequence of layers returned by `_get_encode_layer`.
        """
        encode = nn.Sequential()
        layer_channels = in_channels

        for i, (c, s) in enumerate(zip(channels, strides)):
            layer = self._get_encode_layer(layer_channels, c, s, False)
            encode.add_module("encode_%i" % i, layer)
            layer_channels = c

        return encode, layer_channels

    def _get_intermediate_module(self, in_channels: int, num_inter_units: int) -> Tuple[nn.Module, int]:
        """
        Returns the intermediate block of the network which accepts input from the encoder and whose output goes
        to the decoder.
        """
        # Define some types
        intermediate: nn.Module
        unit: nn.Module

        intermediate = nn.Identity()
        layer_channels = in_channels

        if self.inter_channels:
            intermediate = nn.Sequential()

            for i, (dc, di) in enumerate(zip(self.inter_channels, self.inter_dilations)):
                if self.num_inter_units > 0:
                    unit = ResidualUnit(
                        spatial_dims=self.dimensions,
                        in_channels=layer_channels,
                        out_channels=dc,
                        strides=1,
                        kernel_size=self.kernel_size,
                        subunits=self.num_inter_units,
                        act=self.act,
                        norm=self.norm,
                        dropout=self.dropout,
                        dilation=di,
                        bias=self.bias,
                    )
                else:
                    unit = Convolution(
                        spatial_dims=self.dimensions,
                        in_channels=layer_channels,
                        out_channels=dc,
                        strides=1,
                        kernel_size=self.kernel_size,
                        act=self.act,
                        norm=self.norm,
                        dropout=self.dropout,
                        dilation=di,
                        bias=self.bias,
                    )

                intermediate.add_module("inter_%i" % i, unit)
                layer_channels = dc

        return intermediate, layer_channels

    def _get_decode_module(
        self, in_channels: int, channels: Sequence[int], strides: Sequence[int]
    ) -> Tuple[nn.Sequential, int]:
        """
        Returns the decode part of the network by building up a sequence of layers returned by `_get_decode_layer`.
        """
        decode = nn.Sequential()
        layer_channels = in_channels

        for i, (c, s) in enumerate(zip(channels, strides)):
            layer = self._get_decode_layer(layer_channels, c, s, i == (len(strides) - 1))
            decode.add_module("decode_%i" % i, layer)
            layer_channels = c

        return decode, layer_channels

    def _get_encode_layer(self, in_channels: int, out_channels: int, strides: int, is_last: bool) -> nn.Module:
        """
        Returns a single layer of the encoder part of the network.
        """
        mod: nn.Module
        if self.num_res_units > 0:
            mod = ResidualUnit(
                spatial_dims=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_last,
            )
            return mod
        mod = Convolution(
            spatial_dims=self.dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_last,
        )
        return mod

    def _get_decode_layer(self, in_channels: int, out_channels: int, strides: int, is_last: bool) -> nn.Sequential:
        """
        Returns a single layer of the decoder part of the network.
        """
        decode = nn.Sequential()

        conv = Convolution(
            spatial_dims=self.dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_last and self.num_res_units == 0,
            is_transposed=True,
        )

        decode.add_module("conv", conv)

        if self.num_res_units > 0:
            ru = ResidualUnit(
                spatial_dims=self.dimensions,
                in_channels=out_channels,
                out_channels=out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_last,
            )

            decode.add_module("resunit", ru)

        return decode

    def forward(self, x: torch.Tensor) -> Any:
        x = self.encode(x)
        x = self.intermediate(x)
        x = self.decode(x)
        return x
