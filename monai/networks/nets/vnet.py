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

from typing import Optional, Type, Union

import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Conv, Dropout, Norm


def get_acti_layer(elu: bool, nchan: int):
    if elu:
        elu_type: Type[nn.ELU] = Act[Act.ELU]
        return elu_type(inplace=True)
    else:
        prelu_type: Type[nn.PReLU] = Act[Act.PRELU]
        return prelu_type(nchan)


class LUConv(nn.Module):
    def __init__(self, nchan: int, elu: bool):
        super(LUConv, self).__init__()

        act = ("elu", {"inplace": True}) if elu else ("prelu", {"num_parameters": nchan})
        self.conv_block = Convolution(
            dimensions=3, in_channels=nchan, out_channels=nchan, kernel_size=5, act=act, norm=Norm.BATCH,
        )

    def forward(self, x):
        out = self.conv_block(x)
        return out


def _make_nconv(nchan: int, depth: int, elu: bool):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, inchans: int, outchans: int, elu: bool):
        super(InputTransition, self).__init__()

        if 16 % inchans != 0:
            raise ValueError(f"16 should be divided by inchans, got inchans={inchans}.")

        self.inchans = inchans
        self.relu1 = get_acti_layer(elu, 16)
        self.conv_block = Convolution(
            dimensions=3, in_channels=inchans, out_channels=16, kernel_size=5, act=None, norm=Norm.BATCH,
        )

    def forward(self, x):
        out = self.conv_block(x)
        repeat_num = 16 // self.inchans
        x16 = x.repeat([1, repeat_num, 1, 1, 1])
        out = self.relu1(torch.add(out, x16))
        return out


class DownTransition(nn.Module):
    def __init__(
        self, inchans: int, nconvs: int, elu: bool, dropout_p: Optional[float] = None, dropout_dim: int = 3,
    ):
        super(DownTransition, self).__init__()

        conv3d_type: Type[nn.Conv3d] = Conv[Conv.CONV, 3]
        norm3d_type: Type[nn.BatchNorm3d] = Norm[Norm.BATCH, 3]
        dropout_type: Type[Union[nn.Dropout, nn.Dropout2d, nn.Dropout3d]] = Dropout[Dropout.DROPOUT, dropout_dim]

        outchans = 2 * inchans
        self.down_conv = conv3d_type(inchans, outchans, kernel_size=2, stride=2)
        self.bn1 = norm3d_type(outchans)
        self.relu1 = get_acti_layer(elu, outchans)
        self.relu2 = get_acti_layer(elu, outchans)
        self.ops = _make_nconv(outchans, nconvs, elu)
        self.dropout = dropout_type(dropout_p) if dropout_p is not None else None

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        if self.dropout is not None:
            out = self.dropout(down)
        else:
            out = down
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(
        self,
        inchans: int,
        outchans: int,
        nconvs: int,
        elu: bool,
        dropout_p: Optional[float] = None,
        dropout_dim: int = 3,
    ):
        super(UpTransition, self).__init__()

        conv3d_trans_type: Type[nn.ConvTranspose3d] = Conv[Conv.CONVTRANS, 3]
        norm3d_type: Type[nn.BatchNorm3d] = Norm[Norm.BATCH, 3]
        dropout_type: Type[Union[nn.Dropout, nn.Dropout2d, nn.Dropout3d]] = Dropout[Dropout.DROPOUT, dropout_dim]

        self.up_conv = conv3d_trans_type(inchans, outchans // 2, kernel_size=2, stride=2)
        self.bn1 = norm3d_type(outchans // 2)
        self.dropout = dropout_type(dropout_p) if dropout_p is not None else None
        self.dropout2 = dropout_type(0.5)
        self.relu1 = get_acti_layer(elu, outchans // 2)
        self.relu2 = get_acti_layer(elu, outchans)
        self.ops = _make_nconv(outchans, nconvs, elu)

    def forward(self, x, skipx):
        if self.dropout is not None:
            out = self.dropout(x)
        else:
            out = x
        skipxdo = self.dropout2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inchans: int, outchans: int, elu: bool):
        super(OutputTransition, self).__init__()

        conv3d_type: Type[nn.Conv3d] = Conv[Conv.CONV, 3]
        act = ("elu", {"inplace": True}) if elu else ("prelu", {"num_parameters": outchans})
        self.conv_block = Convolution(
            dimensions=3, in_channels=inchans, out_channels=outchans, kernel_size=5, act=act, norm=Norm.BATCH,
        )
        self.conv2 = conv3d_type(outchans, outchans, kernel_size=1)

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.conv_block(x)
        out = self.conv2(out)
        return out


class VNet(nn.Module):
    """
    V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
    The official Caffe implementation is available in the faustomilletari/VNet repo on GitHub:
    https://github.com/faustomilletari/VNet
    The code is adapted from:
    https://github.com/mattmacy/vnet.pytorch/blob/master/vnet.py

    Args:
        num_input_channel: number of the input image channel. Defaults to 1. The value should meet
            the condition that: 16 % num_input_channel == 0.
        num_output_channel: number of the output image channel. Defaults to 1.
        elu: use ELU activation or PReLU activation. Defaults to False, means using PReLU.
        n_spatial_dim: spatial dimension of input data.
        dropout_p: dropout ratio. Defaults to 0.5.
        dropout_dim: determine the dimensions of dropout. Defaults to 3.
            When dropout_dim = 1, randomly zeroes some of the elements for each channel.
            When dropout_dim = 2, Randomly zero out entire channels (a channel is a 2D feature map).
            When dropout_dim = 3, Randomly zero out entire channels (a channel is a 3D feature map).
    """

    def __init__(
        self,
        num_input_channel: int = 1,
        num_output_channel: int = 1,
        elu: bool = False,
        n_spatial_dim: int = 3,
        dropout_p: float = 0.5,
        dropout_dim: int = 3,
    ):
        super().__init__()

        if n_spatial_dim != 3:
            raise ValueError(f"VNet only supports 3D input, got n_spatial_dim={n_spatial_dim}.")

        self.in_tr = InputTransition(num_input_channel, 16, elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout_p=dropout_p)
        self.down_tr256 = DownTransition(128, 2, elu, dropout_p=dropout_p)
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout_p=dropout_p)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout_p=dropout_p)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, num_output_channel, elu)

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
