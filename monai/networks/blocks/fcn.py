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

from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.utils import exact_version, optional_import

models, _ = optional_import("torchvision", "0.5.0", exact_version, "models")

from monai.networks.layers.factories import Act, Conv, Dropout, Norm


class GCN(nn.Module):
    """
    The Global Convolutional Network module using large 1D
    Kx1 and 1xK kernels to represent 2D kernels. 
    The code is adapted from lsqshr's original version:
    https://github.com/lsqshr/AH-Net/blob/master/net2d.py
    """

    def __init__(self, inplanes: int, planes: int, ks: int = 7):
        """
        Args:
            inplanes: number of input channels.
            planes： number of output channels.
            ks: kernel size for one dimension. Defaults to 7.
        """
        super(GCN, self).__init__()

        conv2d_type: Type[nn.Conv2d] = Conv[Conv.CONV, 2]
        self.conv_l1 = conv2d_type(in_channels=inplanes, out_channels=planes, kernel_size=(ks, 1), padding=(ks // 2, 0))
        self.conv_l2 = conv2d_type(in_channels=planes, out_channels=planes, kernel_size=(1, ks), padding=(0, ks // 2))
        self.conv_r1 = conv2d_type(in_channels=inplanes, out_channels=planes, kernel_size=(1, ks), padding=(0, ks // 2))
        self.conv_r2 = conv2d_type(in_channels=planes, out_channels=planes, kernel_size=(ks, 1), padding=(ks // 2, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: in shape (batch, inplanes, spatial_1, spatial_2).
        """
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x


class Refine(nn.Module):
    """
    Simple residual block to refine the details of the activation maps.
    The code is adapted from lsqshr's original version:
    https://github.com/lsqshr/AH-Net/blob/master/net2d.py
    """

    def __init__(self, planes: int):
        """
        Args:
            planes: number of input channels.
        """
        super(Refine, self).__init__()

        relu_type: Type[nn.ReLU] = Act[Act.RELU]
        conv2d_type: Type[nn.Conv2d] = Conv[Conv.CONV, 2]
        norm2d_type: Type[nn.BatchNorm2d] = Norm[Norm.BATCH, 2]

        self.bn = norm2d_type(num_features=planes)
        self.relu = relu_type(inplace=True)
        self.conv1 = conv2d_type(in_channels=planes, out_channels=planes, kernel_size=3, padding=1)
        self.conv2 = conv2d_type(in_channels=planes, out_channels=planes, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: in shape (batch, planes, spatial_1, spatial_2).
        """
        residual = x
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)

        out = residual + x
        return out


class FCN(nn.Module):
    """
    2D FCN network with 3 input channels. The small decoder is built
    with the GCN and Refine modules.
    The code is adapted from lsqshr's original version:
    https://github.com/lsqshr/AH-Net/blob/master/net2d.py
    """

    def __init__(self, nout: int = 1):
        """
        Args:
            nout: number of output channels. Defaults to 1.
        """
        super(FCN, self).__init__()

        relu_type: Type[nn.ReLU] = Act[Act.RELU]
        conv2d_type: Type[nn.Conv2d] = Conv[Conv.CONV, 2]
        norm2d_type: Type[nn.BatchNorm2d] = Norm[Norm.BATCH, 2]
        dropout_type: Type[nn.Dropout] = Dropout[Dropout.DROPOUT, 1]

        self.relu_type = relu_type
        self.norm2d_type = norm2d_type
        self.conv2d_type = conv2d_type
        self.dropout_type = dropout_type
        self.nout = nout
        resnet = models.resnet50(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn0 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.gcn1 = GCN(2048, self.nout)
        self.gcn2 = GCN(1024, self.nout)
        self.gcn3 = GCN(512, self.nout)
        self.gcn4 = GCN(64, self.nout)
        self.gcn5 = GCN(64, self.nout)

        self.refine1 = Refine(self.nout)
        self.refine2 = Refine(self.nout)
        self.refine3 = Refine(self.nout)
        self.refine4 = Refine(self.nout)
        self.refine5 = Refine(self.nout)
        self.refine6 = Refine(self.nout)
        self.refine7 = Refine(self.nout)
        self.refine8 = Refine(self.nout)
        self.refine9 = Refine(self.nout)
        self.refine10 = Refine(self.nout)
        self.transformer = self.conv2d_type(in_channels=256, out_channels=64, kernel_size=1)

    def _regresser(self, inplanes: int) -> nn.Sequential:

        return nn.Sequential(
            self.conv2d_type(inplanes, inplanes, kernel_size=3, padding=1, bias=False),
            self.norm2d_type(inplanes // 2),
            self.relu_type(inplace=True),
            self.dropout_type(0.1),
            self.conv2d_type(inplanes // 2, self.nout, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: in shape (batch, 3, spatial_1, spatial_2).
        """
        org_input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        gcfm1 = self.refine1(self.gcn1(fm4))
        gcfm2 = self.refine2(self.gcn2(fm3))
        gcfm3 = self.refine3(self.gcn3(fm2))
        gcfm4 = self.refine4(self.gcn4(pool_x))
        gcfm5 = self.refine5(self.gcn5(conv_x))

        fs1 = self.refine6(F.interpolate(gcfm1, fm3.size()[2:], mode="bilinear", align_corners=True) + gcfm2)
        fs2 = self.refine7(F.interpolate(fs1, fm2.size()[2:], mode="bilinear", align_corners=True) + gcfm3)
        fs3 = self.refine8(F.interpolate(fs2, pool_x.size()[2:], mode="bilinear", align_corners=True) + gcfm4)
        fs4 = self.refine9(F.interpolate(fs3, conv_x.size()[2:], mode="bilinear", align_corners=True) + gcfm5)
        out = self.refine10(F.interpolate(fs4, org_input.size()[2:], mode="bilinear", align_corners=True))

        return out


class MCFCN(FCN):
    """
    The multi-channel version of the 2D FCN module.
    Adds a projection layer to take arbitrary number of inputs.
    The code is adapted from lsqshr's original version:
    https://github.com/lsqshr/AH-Net/blob/master/net2d.py
    """

    def __init__(self, nin=3, nout=1):
        """
        Args:
            nin: number of input channels. Defaults to 3.
            nout: number of output channels. Defaults to 1.
        """
        super(MCFCN, self).__init__(nout)

        relu_type: Type[nn.ReLU] = Act[Act.RELU]
        conv2d_type: Type[nn.Conv2d] = Conv[Conv.CONV, 2]
        norm2d_type: Type[nn.BatchNorm2d] = Norm[Norm.BATCH, 2]

        self.init_proj = nn.Sequential(
            conv2d_type(nin, 3, kernel_size=1, padding=0, bias=False), norm2d_type(3), relu_type(inplace=True)
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: in shape (batch, nin, spatial_1, spatial_2).
        """
        x = self.init_proj(x)
        out = super(MCFCN, self).forward(x)
        return out
