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

import math
from typing import Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.layers.factories import Act, Conv, Norm, Pool


class Bottleneck3x3x1(nn.Module):

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: Union[int, Tuple[int, int, int]] = 1,
        downsample: Optional[nn.Sequential] = None,
    ) -> None:

        super(Bottleneck3x3x1, self).__init__()

        conv3d_type: Type[nn.Conv3d] = Conv[Conv.CONV, 3]
        norm3d_type: Type[nn.BatchNorm3d] = Norm[Norm.BATCH, 3]
        relu_type: Type[nn.ReLU] = Act[Act.RELU]
        pool3d_type: Type[nn.MaxPool3d] = Pool[Pool.MAX, 3]

        self.conv1 = conv3d_type(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm3d_type(planes)
        self.conv2 = conv3d_type(planes, planes, kernel_size=(3, 3, 1), stride=stride, padding=(1, 1, 0), bias=False,)
        self.bn2 = norm3d_type(planes)
        self.conv3 = conv3d_type(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm3d_type(planes * 4)
        self.relu = relu_type(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.pool = pool3d_type(kernel_size=(1, 1, 2), stride=(1, 1, 2))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            if out.size() != residual.size():
                out = self.pool(out)

        out += residual
        out = self.relu(out)

        return out


class Projection(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(Projection, self).__init__()

        conv3d_type: Type[nn.Conv3d] = Conv[Conv.CONV, 3]
        norm3d_type: Type[nn.BatchNorm3d] = Norm[Norm.BATCH, 3]
        relu_type: Type[nn.ReLU] = Act[Act.RELU]

        self.add_module("norm", norm3d_type(num_input_features))
        self.add_module("relu", relu_type(inplace=True))
        self.add_module("conv", nn.Conv3d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))


class DenseBlock(nn.Sequential):
    def __init__(self, num_layers: int, num_input_features: int, bn_size: int, growth_rate: int, drop_rate: float):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = Pseudo3DLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module("denselayer%d" % (i + 1), layer)


class UpTransition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int, upsample_mode: str = "transpose"):
        super(UpTransition, self).__init__()

        conv3d_type: Type[nn.Conv3d] = Conv[Conv.CONV, 3]
        norm3d_type: Type[nn.BatchNorm3d] = Norm[Norm.BATCH, 3]
        relu_type: Type[nn.ReLU] = Act[Act.RELU]

        self.add_module("norm", norm3d_type(num_input_features))
        self.add_module("relu", relu_type(inplace=True))
        self.add_module(
            "conv", conv3d_type(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        )
        if upsample_mode == "transpose":
            conv3d_trans_type: Type[nn.ConvTranspose3d] = Conv[Conv.CONVTRANS, 3]
            self.add_module(
                "pool", conv3d_trans_type(num_output_features, num_output_features, kernel_size=2, stride=2, bias=False)
            )
        elif upsample_mode == "interpolate":
            self.add_module("pool", nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True))
        else:
            raise NotImplementedError(
                f"Currently only 'transpose' and 'interpolate' modes are supported, got {upsample_mode}."
            )


class Final(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int, upsample_mode: str = "transpose"):
        super(Final, self).__init__()

        conv3d_type: Type[nn.Conv3d] = Conv[Conv.CONV, 3]
        norm3d_type: Type[nn.BatchNorm3d] = Norm[Norm.BATCH, 3]
        relu_type: Type[nn.ReLU] = Act[Act.RELU]

        self.add_module("norm", norm3d_type(num_input_features))
        self.add_module("relu", relu_type(inplace=True))
        self.add_module(
            "conv",
            conv3d_type(
                num_input_features, num_output_features, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0), bias=False
            ),
        )
        if upsample_mode == "transpose":
            conv3d_trans_type: Type[nn.ConvTranspose3d] = Conv[Conv.CONVTRANS, 3]
            self.add_module(
                "up", conv3d_trans_type(num_output_features, num_output_features, kernel_size=2, stride=2, bias=False)
            )
        elif upsample_mode == "interpolate":
            self.add_module("up", nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True))
        else:
            raise NotImplementedError(
                f"Currently only 'transpose' and 'interpolate' modes are supported, got {upsample_mode}."
            )


class Pseudo3DLayer(nn.Module):
    def __init__(self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float):
        super(Pseudo3DLayer, self).__init__()
        # 1x1x1

        conv3d_type: Type[nn.Conv3d] = Conv[Conv.CONV, 3]
        norm3d_type: Type[nn.BatchNorm3d] = Norm[Norm.BATCH, 3]
        relu_type: Type[nn.ReLU] = Act[Act.RELU]

        self.bn1 = norm3d_type(num_input_features)
        self.relu1 = relu_type(inplace=True)
        self.conv1 = conv3d_type(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        # 3x3x1
        self.bn2 = norm3d_type(bn_size * growth_rate)
        self.relu2 = relu_type(inplace=True)
        self.conv2 = conv3d_type(
            bn_size * growth_rate, growth_rate, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0), bias=False
        )
        # 1x1x3
        self.bn3 = norm3d_type(growth_rate)
        self.relu3 = relu_type(inplace=True)
        self.conv3 = conv3d_type(
            growth_rate, growth_rate, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1), bias=False
        )
        # 1x1x1
        self.bn4 = norm3d_type(growth_rate)
        self.relu4 = relu_type(inplace=True)
        self.conv4 = conv3d_type(growth_rate, growth_rate, kernel_size=1, stride=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        inx = x
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.relu2(x)
        x3x3x1 = self.conv2(x)

        x = self.bn3(x3x3x1)
        x = self.relu3(x)
        x1x1x3 = self.conv3(x)

        x = x3x3x1 + x1x1x3
        x = self.bn4(x)
        x = self.relu4(x)
        new_features = self.conv4(x)

        self.drop_rate = 0  # Dropout will make trouble!
        # since we use the train mode for inference
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([inx, new_features], 1)


class PSP(nn.Module):
    def __init__(self, in_ch: int, upsample_mode: str = "transpose"):
        super(PSP, self).__init__()

        conv3d_type: Type[nn.Conv3d] = Conv[Conv.CONV, 3]
        pool3d_type: Type[nn.MaxPool3d] = Pool[Pool.MAX, 3]

        self.pool64 = pool3d_type(kernel_size=(64, 64, 1), stride=(64, 64, 1))
        self.pool32 = pool3d_type(kernel_size=(32, 32, 1), stride=(32, 32, 1))
        self.pool16 = pool3d_type(kernel_size=(16, 16, 1), stride=(16, 16, 1))
        self.pool8 = pool3d_type(kernel_size=(8, 8, 1), stride=(8, 8, 1))

        self.proj64 = conv3d_type(in_ch, 1, kernel_size=(1, 1, 1), stride=1, padding=(1, 1, 0))
        self.proj32 = conv3d_type(in_ch, 1, kernel_size=(1, 1, 1), stride=1, padding=(1, 1, 0))
        self.proj16 = conv3d_type(in_ch, 1, kernel_size=(1, 1, 1), stride=1, padding=(1, 1, 0))
        self.proj8 = conv3d_type(in_ch, 1, kernel_size=(1, 1, 1), stride=1, padding=(1, 1, 0))

        self.upsample_mode = upsample_mode
        if self.upsample_mode == "transpose":
            conv3d_trans_type: Type[nn.ConvTranspose3d] = Conv[Conv.CONVTRANS, 3]
            self.up64 = conv3d_trans_type(1, 1, kernel_size=(64, 64, 1), stride=(64, 64, 1), padding=(64, 64, 0))
            self.up32 = conv3d_trans_type(1, 1, kernel_size=(32, 32, 1), stride=(32, 32, 1), padding=(32, 32, 0))
            self.up16 = conv3d_trans_type(1, 1, kernel_size=(16, 16, 1), stride=(16, 16, 1), padding=(16, 16, 0))
            self.up8 = conv3d_trans_type(1, 1, kernel_size=(8, 8, 1), stride=(8, 8, 1), padding=(8, 8, 0))

    def forward(self, x):
        if self.upsample_mode == "transpose":
            x64 = self.up64(self.proj64(self.pool64(x)))
            x32 = self.up32(self.proj32(self.pool32(x)))
            x16 = self.up16(self.proj16(self.pool16(x)))
            x8 = self.up8(self.proj8(self.pool8(x)))
        elif self.upsample_mode == "interpolate":
            x64 = F.interpolate(
                self.proj64(self.pool64(x)),
                size=(x.size(2), x.size(3), x.size(4)),
                mode="trilinear",
                align_corners=True,
            )
            x32 = F.interpolate(
                self.proj32(self.pool32(x)),
                size=(x.size(2), x.size(3), x.size(4)),
                mode="trilinear",
                align_corners=True,
            )
            x16 = F.interpolate(
                self.proj16(self.pool16(x)),
                size=(x.size(2), x.size(3), x.size(4)),
                mode="trilinear",
                align_corners=True,
            )
            x8 = F.interpolate(
                self.proj8(self.pool8(x)), size=(x.size(2), x.size(3), x.size(4)), mode="trilinear", align_corners=True
            )
        else:
            raise NotImplementedError(
                f"Currently only 'transpose' and 'interpolate' modes are supported, got {self.upsample_mode}."
            )
        x = torch.cat((x64, x32, x16, x8), dim=1)
        return x


class AHNet(nn.Module):
    """
    Anisotropic Hybrid Network (AH-Net).
    The code is adapted from lsqshr's original version:
    https://github.com/lsqshr/AH-Net/blob/master/net3d.py
    In order to use pretrained weights from 2D FCN/MCFCN, please call the copy_from function, for example:
    ahnet = AHNet(upsample_mode='transpose')
    ahnet.copy_from(model_2d)

    Args:
        layers (list): number of residual blocks for 4 layers of the network (layer1...layer4).
        upsample_mode (str): The mode of upsampling manipulations, there are two choices:
            1) "transpose", uses transposed convolution layers.
            2) "interpolate", uses standard interpolate way.
            Using the second mode cannot guarantee the model's reproducibility. Defaults to "transpose".
    """

    def __init__(self, layers: tuple = (3, 4, 6, 3), upsample_mode: str = "transpose"):
        self.inplanes = 64
        super(AHNet, self).__init__()

        conv2d_type: Type[nn.Conv2d] = Conv[Conv.CONV, 2]
        conv3d_type: Type[nn.Conv3d] = Conv[Conv.CONV, 3]
        conv3d_trans_type: Type[nn.ConvTranspose3d] = Conv[Conv.CONVTRANS, 3]
        norm2d_type: Type[nn.BatchNorm2d] = Norm[Norm.BATCH, 2]
        norm3d_type: Type[nn.BatchNorm3d] = Norm[Norm.BATCH, 3]
        relu_type: Type[nn.ReLU] = Act[Act.RELU]
        pool3d_type: Type[nn.MaxPool3d] = Pool[Pool.MAX, 3]

        self.conv2d_type = conv2d_type
        self.conv3d_type = conv3d_type
        self.norm2d_type = norm2d_type
        self.norm3d_type = norm3d_type
        self.relu_type = relu_type
        self.pool3d_type = pool3d_type
        # Make the 3x3x1 resnet layers
        self.conv1 = conv3d_type(1, 64, kernel_size=(7, 7, 3), stride=(2, 2, 1), padding=(3, 3, 1), bias=False)
        self.pool1 = pool3d_type(kernel_size=(1, 1, 2), stride=(1, 1, 2))
        self.bn0 = norm3d_type(64)
        self.relu = relu_type(inplace=True)
        if upsample_mode == "transpose":
            self.maxpool = pool3d_type(kernel_size=(2, 2, 2), stride=2)
        elif upsample_mode == "interpolate":
            self.maxpool = pool3d_type(kernel_size=(3, 3, 3), stride=2, padding=1)
        else:
            raise NotImplementedError(
                f"Currently only 'transpose' and 'interpolate' modes are supported, got {upsample_mode}."
            )

        self.layer1 = self._make_layer(Bottleneck3x3x1, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(Bottleneck3x3x1, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck3x3x1, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(Bottleneck3x3x1, 512, layers[3], stride=2)

        # Make the 3D dense decoder layers
        densegrowth = 20
        densebn = 4
        ndenselayer = 3

        num_init_features = 64
        noutres1 = 256
        noutres2 = 512
        noutres3 = 1024
        noutres4 = 2048

        self.up0 = UpTransition(noutres4, noutres3, upsample_mode)
        self.dense0 = DenseBlock(
            num_layers=ndenselayer, num_input_features=noutres3, bn_size=densebn, growth_rate=densegrowth, drop_rate=0.0
        )
        noutdense = noutres3 + ndenselayer * densegrowth

        self.up1 = UpTransition(noutdense, noutres2, upsample_mode)
        self.dense1 = DenseBlock(
            num_layers=ndenselayer, num_input_features=noutres2, bn_size=densebn, growth_rate=densegrowth, drop_rate=0.0
        )
        noutdense1 = noutres2 + ndenselayer * densegrowth

        self.up2 = UpTransition(noutdense1, noutres1, upsample_mode)
        self.dense2 = DenseBlock(
            num_layers=ndenselayer, num_input_features=noutres1, bn_size=densebn, growth_rate=densegrowth, drop_rate=0.0
        )
        noutdense2 = noutres1 + ndenselayer * densegrowth

        self.trans1 = Projection(noutdense2, num_init_features)
        self.dense3 = DenseBlock(
            num_layers=ndenselayer,
            num_input_features=num_init_features,
            bn_size=densebn,
            growth_rate=densegrowth,
            drop_rate=0.0,
        )
        noutdense3 = num_init_features + densegrowth * ndenselayer

        self.up3 = UpTransition(noutdense3, num_init_features, upsample_mode)
        self.dense4 = DenseBlock(
            num_layers=ndenselayer,
            num_input_features=num_init_features,
            bn_size=densebn,
            growth_rate=densegrowth,
            drop_rate=0.0,
        )
        noutdense4 = num_init_features + densegrowth * ndenselayer

        self.psp = PSP(noutdense4, upsample_mode)
        self.final = Final(4 + noutdense4, 1, upsample_mode)

        # Initialise parameters
        for m in self.modules():
            if isinstance(m, (conv3d_type, conv3d_trans_type)):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, norm3d_type):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block: Type[Bottleneck3x3x1], planes: int, blocks: int, stride: int = 1,) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.conv3d_type(
                    self.inplanes, planes * block.expansion, kernel_size=1, stride=(stride, stride, 1), bias=False
                ),
                self.pool3d_type(kernel_size=(1, 1, stride), stride=(1, 1, stride)),
                self.norm3d_type(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, (stride, stride, 1), downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        sum0 = self.up0(fm4) + fm3
        d0 = self.dense0(sum0)

        sum1 = self.up1(d0) + fm2
        d1 = self.dense1(sum1)

        sum2 = self.up2(d1) + fm1
        d2 = self.dense2(sum2)

        sum3 = self.trans1(d2) + pool_x
        d3 = self.dense3(sum3)

        sum4 = self.up3(d3) + conv_x
        d4 = self.dense4(sum4)

        psp = self.psp(d4)
        x = torch.cat((psp, d4), dim=1)
        return self.final(x)

    def copy_from(self, net):
        # Copy the initial module CONV1 -- Need special care since
        # we only have one input channel in the 3D network
        p2d, p3d = next(net.conv1.parameters()), next(self.conv1.parameters())

        # From 64x3x7x7 -> 64x3x7x7x1 -> 64x1x7x7x3
        p3d.data = p2d.data.unsqueeze(dim=4).permute(0, 4, 2, 3, 1).clone()

        # Copy the initial module BN0
        copy_bn_param(net.bn0, self.bn0)

        # Copy layer1 to layer4
        for i in range(1, 5):
            layer_num = "layer" + str(i)

            layer_2d = []
            layer_3d = []
            for m1 in vars(net)["_modules"][layer_num].modules():
                if isinstance(m1, (self.norm2d_type, self.conv2d_type)):
                    layer_2d.append(m1)
            for m2 in vars(self)["_modules"][layer_num].modules():
                if isinstance(m2, (self.norm3d_type, self.conv3d_type)):
                    layer_3d.append(m2)

            for m1, m2 in zip(layer_2d, layer_3d):
                if isinstance(m1, self.conv2d_type):
                    copy_conv_param(m1, m2)
                if isinstance(m1, self.norm2d_type):
                    copy_bn_param(m1, m2)


def copy_conv_param(module2d, module3d):
    for p2d, p3d in zip(module2d.parameters(), module3d.parameters()):
        p3d.data[:] = p2d.data.unsqueeze(dim=4).clone()[:]


def copy_bn_param(module2d, module3d):
    for p2d, p3d in zip(module2d.parameters(), module3d.parameters()):
        p3d.data[:] = p2d.data[:]  # Two parameter gamma and beta
