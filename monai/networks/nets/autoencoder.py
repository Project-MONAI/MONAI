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

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm


class AutoEncoder(nn.Module):
    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        inter_channels: list = list(),
        inter_dilations: list = list(),
        num_inter_units: int = 2,
        act: Optional[Union[Tuple, str]] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: Optional[Union[Tuple, str, float]] = None,
    ) -> None:

        super().__init__()
        self.dimensions = dimensions
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
        self.num_inter_units = num_inter_units
        self.inter_channels = list(inter_channels)
        self.inter_dilations = list(inter_dilations or [1] * len(inter_channels))

        self.encoded_channels = in_channels
        decode_channel_list = list(channels[-2::-1]) + [out_channels]

        self.encode, self.encoded_channels = self._get_encode_module(self.encoded_channels, channels, strides)
        self.intermediate, self.encoded_channels = self._get_intermediate_module(self.encoded_channels, num_inter_units)
        self.decode, _ = self._get_decode_module(self.encoded_channels, decode_channel_list, strides[::-1] or [1])

    def _get_encode_module(
        self, in_channels: int, channels: Sequence[int], strides: Sequence[int]
    ) -> (nn.Sequential, int):
        encode = nn.Sequential()
        layer_channels = in_channels

        for i, (c, s) in enumerate(zip(channels, strides)):
            layer = self._get_encode_layer(layer_channels, c, s, False)
            encode.add_module("encode_%i" % i, layer)
            layer_channels = c

        return encode, layer_channels

    def _get_intermediate_module(self, in_channels: int, num_inter_units: int) -> (nn.Module, int):
        intermediate = nn.Identity()
        layer_channels = in_channels

        if self.inter_channels:
            intermediate = nn.Sequential()

            for i, (dc, di) in enumerate(zip(self.inter_channels, self.inter_dilations)):
                if self.num_inter_units > 0:
                    unit = ResidualUnit(
                        self.dimensions,
                        layer_channels,
                        dc,
                        1,
                        self.kernel_size,
                        self.num_inter_units,
                        self.act,
                        self.norm,
                        self.dropout,
                        di,
                    )
                else:
                    unit = Convolution(
                        self.dimensions, layer_channels, dc, 1, self.kernel_size, self.act, self.norm, self.dropout, di
                    )

                intermediate.add_module("inter_%i" % i, unit)
                layer_channels = dc

        return intermediate, layer_channels

    def _get_decode_module(self, in_channels: int, channels: int, strides: Sequence[int]) -> (nn.Sequential, int):
        decode = nn.Sequential()
        layer_channels = in_channels

        for i, (c, s) in enumerate(zip(channels, strides)):
            layer = self._get_decode_layer(layer_channels, c, s, i == (len(strides) - 1))
            decode.add_module("decode_%i" % i, layer)
            layer_channels = c

        return decode, layer_channels

    def _get_encode_layer(
        self, in_channels: int, out_channels: int, strides: Sequence[int], is_last: bool
    ) -> nn.Module:

        if self.num_res_units > 0:
            return ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides,
                self.kernel_size,
                self.num_res_units,
                self.act,
                self.norm,
                self.dropout,
                last_conv_only=is_last,
            )
        else:
            return Convolution(
                self.dimensions,
                in_channels,
                out_channels,
                strides,
                self.kernel_size,
                self.act,
                self.norm,
                self.dropout,
                conv_only=is_last,
            )

    def _get_decode_layer(
        self, in_channels: int, out_channels: int, strides: Sequence[int], is_last: bool
    ) -> nn.Sequential:

        decode = nn.Sequential()

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides,
            self.up_kernel_size,
            self.act,
            self.norm,
            self.dropout,
            conv_only=is_last and self.num_res_units == 0,
            is_transposed=True,
        )

        decode.add_module("conv", conv)

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                1,
                self.kernel_size,
                1,
                self.act,
                self.norm,
                self.dropout,
                last_conv_only=is_last,
            )

            decode.add_module("resunit", ru)

        return decode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        x = self.intermediate(x)
        x = self.decode(x)
        return x
