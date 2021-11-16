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

import numpy as np
import torch
import torch.nn as nn


class FactorizedReduce(nn.Module):
    """
    Down-sampling the feature by 2 using stride.
    """
    def __init__(
        self,
        C_in: int,
        C_out: int,
    ):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv3d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv3d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.InstanceNorm3d(C_out)

        # multiply by 8 to comply with cell output size (see net.get_memory_usage)
        # self.memory = (1 + C_out/C_in/8 * 3) * 8 * C_in/C_out

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:,1:])], dim=1)
        out = self.bn(out)
        return out


class FactorizedIncrease(nn.Module):
    """
    Up-sampling the feature by 2 using stride.
    """
    def __init__ (
        self,
        in_channel: int,
        out_channel: int,
    ):
        super(FactorizedIncrease, self).__init__()
        self._in_channel = in_channel
        self.op = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True),
            nn.ReLU(),
            nn.Conv3d(self._in_channel, out_channel, 1, stride=1, padding=0, bias=False),
            nn.InstanceNorm3d(out_channel)
        )

        # devide by 8 to comply with cell output size
        # self.memory = 8*(1+1+out_channel/in_channel*2)/8 * in_channel/out_channel

    def forward (self, x):
        return self.op(x)


class P3DReLUConvBN(nn.Module):
    def __init__(
        self,
        C_in: int,
        C_out: int,
        kernel_size: int,
        padding: int,
        P3Dmode: int = 0
    ):
        super().__init__()
        self.P3Dmode = P3Dmode
        if P3Dmode == 0: #331
            kernel_size0 = (kernel_size, kernel_size, 1)
            kernel_size1 = (1, 1, kernel_size)
            padding0 = (padding, padding, 0)
            padding1 = (0, 0, padding)
        elif P3Dmode == 1: #313
            kernel_size0 = (kernel_size, 1, kernel_size)
            kernel_size1 = (1, kernel_size, 1)
            padding0 = (padding, 0, padding)
            padding1 = (0, padding, 0)
        elif P3Dmode == 2:
            kernel_size0 = (1, kernel_size, kernel_size)
            kernel_size1 = (kernel_size, 1, 1)
            padding0 = (0, padding, padding)
            padding1 = (padding, 0, 0)

        self.op = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(C_in, C_in, kernel_size0,
                        padding=padding0, bias=False),
            nn.Conv3d(C_in, C_out, kernel_size1,
                        padding=padding1, bias=False),
            nn.InstanceNorm3d(C_out)
        )
        self.memory = 1 + 1 + C_out/C_in * 2

    def forward(self, x):
        return self.op(x)


class ReLUConvBN(nn.Module):
    def __init__(
        self,
        C_in: int,
        C_out: int,
        kernel_size: int,
        padding: int
    ):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(C_in, C_out, kernel_size, padding=padding, bias=False),
            nn.InstanceNorm3d(C_out)
        )
        # self.memory = 1 + C_out/C_in * 2

    def forward(self, x):
        return self.op(x)