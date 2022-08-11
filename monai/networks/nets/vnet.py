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

from typing import Dict, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Conv, Dropout, Norm, split_args

__all__ = ["VNet"]


def get_acti_layer(act: Union[Tuple[str, Dict], str], nchan: int = 0):
    if act == "prelu":
        act = ("prelu", {"num_parameters": nchan})
    act_name, act_args = split_args(act)
    act_type = Act[act_name]
    return act_type(**act_args)


class LUConv(nn.Module):
    def __init__(self, spatial_dims: int, nchan: int, act: Union[Tuple[str, Dict], str], bias: bool = False):
        super().__init__()

        self.act_function = get_acti_layer(act, nchan)
        self.conv_block = Convolution(
            spatial_dims=spatial_dims,
            in_channels=nchan,
            out_channels=nchan,
            kernel_size=5,
            act=None,
            norm=Norm.BATCH,
            bias=bias,
        )

    def forward(self, x):
        out = self.conv_block(x)
        out = self.act_function(out)
        return out


def _make_nconv(spatial_dims: int, nchan: int, depth: int, act: Union[Tuple[str, Dict], str], bias: bool = False):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(spatial_dims, nchan, act, bias))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        act: Union[Tuple[str, Dict], str],
        bias: bool = False,
    ):
        super().__init__()

        if out_channels % in_channels != 0:
            raise ValueError(f"16 should be divisible by in_channels, got in_channels={in_channels}.")

        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_function = get_acti_layer(act, out_channels)
        self.conv_block = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5,
            act=None,
            norm=Norm.BATCH,
            bias=bias,
        )

    def forward(self, x):
        out = self.conv_block(x)
        repeat_num = self.out_channels // self.in_channels
        x16 = x.repeat([1, repeat_num, 1, 1, 1][: self.spatial_dims + 2])
        out = self.act_function(torch.add(out, x16))
        return out


class DownTransition(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        nconvs: int,
        act: Union[Tuple[str, Dict], str],
        dropout_prob: Optional[float] = None,
        dropout_dim: int = 3,
        bias: bool = False,
    ):
        super().__init__()

        conv_type: Type[Union[nn.Conv2d, nn.Conv3d]] = Conv[Conv.CONV, spatial_dims]
        norm_type: Type[Union[nn.BatchNorm2d, nn.BatchNorm3d]] = Norm[Norm.BATCH, spatial_dims]
        dropout_type: Type[Union[nn.Dropout, nn.Dropout2d, nn.Dropout3d]] = Dropout[Dropout.DROPOUT, dropout_dim]

        out_channels = 2 * in_channels
        self.down_conv = conv_type(in_channels, out_channels, kernel_size=2, stride=2, bias=bias)
        self.bn1 = norm_type(out_channels)
        self.act_function1 = get_acti_layer(act, out_channels)
        self.act_function2 = get_acti_layer(act, out_channels)
        self.ops = _make_nconv(spatial_dims, out_channels, nconvs, act, bias)
        self.dropout = dropout_type(dropout_prob) if dropout_prob is not None else None

    def forward(self, x):
        down = self.act_function1(self.bn1(self.down_conv(x)))
        if self.dropout is not None:
            out = self.dropout(down)
        else:
            out = down
        out = self.ops(out)
        out = self.act_function2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        nconvs: int,
        act: Union[Tuple[str, Dict], str],
        dropout_prob: Optional[float] = None,
        dropout_dim: int = 3,
    ):
        super().__init__()

        conv_trans_type: Type[Union[nn.ConvTranspose2d, nn.ConvTranspose3d]] = Conv[Conv.CONVTRANS, spatial_dims]
        norm_type: Type[Union[nn.BatchNorm2d, nn.BatchNorm3d]] = Norm[Norm.BATCH, spatial_dims]
        dropout_type: Type[Union[nn.Dropout, nn.Dropout2d, nn.Dropout3d]] = Dropout[Dropout.DROPOUT, dropout_dim]

        self.up_conv = conv_trans_type(in_channels, out_channels // 2, kernel_size=2, stride=2)
        self.bn1 = norm_type(out_channels // 2)
        self.dropout = dropout_type(dropout_prob) if dropout_prob is not None else None
        self.dropout2 = dropout_type(0.5)
        self.act_function1 = get_acti_layer(act, out_channels // 2)
        self.act_function2 = get_acti_layer(act, out_channels)
        self.ops = _make_nconv(spatial_dims, out_channels, nconvs, act)

    def forward(self, x, skipx):
        if self.dropout is not None:
            out = self.dropout(x)
        else:
            out = x
        skipxdo = self.dropout2(skipx)
        out = self.act_function1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.act_function2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        act: Union[Tuple[str, Dict], str],
        bias: bool = False,
    ):
        super().__init__()

        conv_type: Type[Union[nn.Conv2d, nn.Conv3d]] = Conv[Conv.CONV, spatial_dims]

        self.act_function1 = get_acti_layer(act, out_channels)
        self.conv_block = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5,
            act=None,
            norm=Norm.BATCH,
            bias=bias,
        )
        self.conv2 = conv_type(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.conv_block(x)
        out = self.act_function1(out)
        out = self.conv2(out)
        return out


class VNet(nn.Module):
    """
    V-Net based on `Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
    <https://arxiv.org/pdf/1606.04797.pdf>`_.
    Adapted from `the official Caffe implementation
    <https://github.com/faustomilletari/VNet>`_. and `another pytorch implementation
    <https://github.com/mattmacy/vnet.pytorch/blob/master/vnet.py>`_.
    The model supports 2D or 3D inputs.

    Args:
        spatial_dims: spatial dimension of the input data. Defaults to 3.
        in_channels: number of input channels for the network. Defaults to 1.
            The value should meet the condition that ``16 % in_channels == 0``.
        out_channels: number of output channels for the network. Defaults to 1.
        act: activation type in the network. Defaults to ``("elu", {"inplace": True})``.
        dropout_prob: dropout ratio. Defaults to 0.5.
        dropout_dim: determine the dimensions of dropout. Defaults to 3.

            - ``dropout_dim = 1``, randomly zeroes some of the elements for each channel.
            - ``dropout_dim = 2``, Randomly zeroes out entire channels (a channel is a 2D feature map).
            - ``dropout_dim = 3``, Randomly zeroes out entire channels (a channel is a 3D feature map).
        bias: whether to have a bias term in convolution blocks. Defaults to False.
            According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
            if a conv layer is directly followed by a batch norm layer, bias should be False.

    """

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        act: Union[Tuple[str, Dict], str] = ("elu", {"inplace": True}),
        dropout_prob: float = 0.5,
        dropout_dim: int = 3,
        bias: bool = False,
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise AssertionError("spatial_dims can only be 2 or 3.")

        self.in_tr = InputTransition(spatial_dims, in_channels, 16, act, bias=bias)
        self.down_tr32 = DownTransition(spatial_dims, 16, 1, act, bias=bias)
        self.down_tr64 = DownTransition(spatial_dims, 32, 2, act, bias=bias)
        self.down_tr128 = DownTransition(spatial_dims, 64, 3, act, dropout_prob=dropout_prob, bias=bias)
        self.down_tr256 = DownTransition(spatial_dims, 128, 2, act, dropout_prob=dropout_prob, bias=bias)
        self.up_tr256 = UpTransition(spatial_dims, 256, 256, 2, act, dropout_prob=dropout_prob)
        self.up_tr128 = UpTransition(spatial_dims, 256, 128, 2, act, dropout_prob=dropout_prob)
        self.up_tr64 = UpTransition(spatial_dims, 128, 64, 1, act)
        self.up_tr32 = UpTransition(spatial_dims, 64, 32, 1, act)
        self.out_tr = OutputTransition(spatial_dims, 32, out_channels, act, bias=bias)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        x = self.up_tr256(out256, out128)
        x = self.up_tr128(x, out64)
        x = self.up_tr64(x, out32)
        x = self.up_tr32(x, out16)
        x = self.out_tr(x)
        return x
