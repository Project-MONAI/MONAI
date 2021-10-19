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

import math
from typing import Optional, Sequence, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.fcn import FCN
from monai.networks.layers.factories import Act, Conv, Norm, Pool

__all__ = ["AHnet", "Ahnet", "AHNet"]


class Bottleneck3x3x1(nn.Module):

    expansion = 4

    def __init__(
        self,
        spatial_dims: int,
        inplanes: int,
        planes: int,
        stride: Union[Sequence[int], int] = 1,
        downsample: Optional[nn.Sequential] = None,
    ) -> None:

        super().__init__()

        conv_type = Conv[Conv.CONV, spatial_dims]
        norm_type: Type[Union[nn.BatchNorm2d, nn.BatchNorm3d]] = Norm[Norm.BATCH, spatial_dims]
        pool_type: Type[Union[nn.MaxPool2d, nn.MaxPool3d]] = Pool[Pool.MAX, spatial_dims]
        relu_type: Type[nn.ReLU] = Act[Act.RELU]

        self.conv1 = conv_type(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_type(planes)
        self.conv2 = conv_type(
            planes,
            planes,
            kernel_size=(3, 3, 1)[-spatial_dims:],
            stride=stride,
            padding=(1, 1, 0)[-spatial_dims:],
            bias=False,
        )
        self.bn2 = norm_type(planes)
        self.conv3 = conv_type(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_type(planes * 4)
        self.relu = relu_type(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.pool = pool_type(kernel_size=(1, 1, 2)[-spatial_dims:], stride=(1, 1, 2)[-spatial_dims:])

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
    def __init__(self, spatial_dims: int, num_input_features: int, num_output_features: int):
        super().__init__()

        conv_type = Conv[Conv.CONV, spatial_dims]
        norm_type: Type[Union[nn.BatchNorm2d, nn.BatchNorm3d]] = Norm[Norm.BATCH, spatial_dims]
        relu_type: Type[nn.ReLU] = Act[Act.RELU]

        self.add_module("norm", norm_type(num_input_features))
        self.add_module("relu", relu_type(inplace=True))
        self.add_module("conv", conv_type(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))


class DenseBlock(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        dropout_prob: float,
    ):
        super().__init__()
        for i in range(num_layers):
            layer = Pseudo3DLayer(
                spatial_dims, num_input_features + i * growth_rate, growth_rate, bn_size, dropout_prob
            )
            self.add_module("denselayer%d" % (i + 1), layer)


class UpTransition(nn.Sequential):
    def __init__(
        self, spatial_dims: int, num_input_features: int, num_output_features: int, upsample_mode: str = "transpose"
    ):
        super().__init__()

        conv_type = Conv[Conv.CONV, spatial_dims]
        norm_type: Type[Union[nn.BatchNorm2d, nn.BatchNorm3d]] = Norm[Norm.BATCH, spatial_dims]
        relu_type: Type[nn.ReLU] = Act[Act.RELU]

        self.add_module("norm", norm_type(num_input_features))
        self.add_module("relu", relu_type(inplace=True))
        self.add_module("conv", conv_type(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        if upsample_mode == "transpose":
            conv_trans_type = Conv[Conv.CONVTRANS, spatial_dims]
            self.add_module(
                "up", conv_trans_type(num_output_features, num_output_features, kernel_size=2, stride=2, bias=False)
            )
        else:
            align_corners: Optional[bool] = None
            if upsample_mode in ["trilinear", "bilinear"]:
                align_corners = True
            self.add_module("up", nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=align_corners))


class Final(nn.Sequential):
    def __init__(
        self, spatial_dims: int, num_input_features: int, num_output_features: int, upsample_mode: str = "transpose"
    ):
        super().__init__()

        conv_type = Conv[Conv.CONV, spatial_dims]
        norm_type: Type[Union[nn.BatchNorm2d, nn.BatchNorm3d]] = Norm[Norm.BATCH, spatial_dims]
        relu_type: Type[nn.ReLU] = Act[Act.RELU]

        self.add_module("norm", norm_type(num_input_features))
        self.add_module("relu", relu_type(inplace=True))
        self.add_module(
            "conv",
            conv_type(
                num_input_features,
                num_output_features,
                kernel_size=(3, 3, 1)[-spatial_dims:],
                stride=1,
                padding=(1, 1, 0)[-spatial_dims:],
                bias=False,
            ),
        )
        if upsample_mode == "transpose":
            conv_trans_type = Conv[Conv.CONVTRANS, spatial_dims]
            self.add_module(
                "up", conv_trans_type(num_output_features, num_output_features, kernel_size=2, stride=2, bias=False)
            )
        else:
            align_corners: Optional[bool] = None
            if upsample_mode in ["trilinear", "bilinear"]:
                align_corners = True
            self.add_module("up", nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=align_corners))


class Pseudo3DLayer(nn.Module):
    def __init__(self, spatial_dims: int, num_input_features: int, growth_rate: int, bn_size: int, dropout_prob: float):
        super().__init__()
        # 1x1x1

        conv_type = Conv[Conv.CONV, spatial_dims]
        norm_type: Type[Union[nn.BatchNorm2d, nn.BatchNorm3d]] = Norm[Norm.BATCH, spatial_dims]
        relu_type: Type[nn.ReLU] = Act[Act.RELU]

        self.bn1 = norm_type(num_input_features)
        self.relu1 = relu_type(inplace=True)
        self.conv1 = conv_type(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        # 3x3x1
        self.bn2 = norm_type(bn_size * growth_rate)
        self.relu2 = relu_type(inplace=True)
        self.conv2 = conv_type(
            bn_size * growth_rate,
            growth_rate,
            kernel_size=(3, 3, 1)[-spatial_dims:],
            stride=1,
            padding=(1, 1, 0)[-spatial_dims:],
            bias=False,
        )
        # 1x1x3
        self.bn3 = norm_type(growth_rate)
        self.relu3 = relu_type(inplace=True)
        self.conv3 = conv_type(
            growth_rate,
            growth_rate,
            kernel_size=(1, 1, 3)[-spatial_dims:],
            stride=1,
            padding=(0, 0, 1)[-spatial_dims:],
            bias=False,
        )
        # 1x1x1
        self.bn4 = norm_type(growth_rate)
        self.relu4 = relu_type(inplace=True)
        self.conv4 = conv_type(growth_rate, growth_rate, kernel_size=1, stride=1, bias=False)
        self.dropout_prob = dropout_prob

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

        self.dropout_prob = 0.0  # Dropout will make trouble!
        # since we use the train mode for inference
        if self.dropout_prob > 0.0:
            new_features = F.dropout(new_features, p=self.dropout_prob, training=self.training)
        return torch.cat([inx, new_features], 1)


class PSP(nn.Module):
    def __init__(self, spatial_dims: int, psp_block_num: int, in_ch: int, upsample_mode: str = "transpose"):
        super().__init__()
        self.up_modules = nn.ModuleList()
        conv_type = Conv[Conv.CONV, spatial_dims]
        pool_type: Type[Union[nn.MaxPool2d, nn.MaxPool3d]] = Pool[Pool.MAX, spatial_dims]

        self.pool_modules = nn.ModuleList()
        self.project_modules = nn.ModuleList()

        for i in range(psp_block_num):
            size = (2 ** (i + 3), 2 ** (i + 3), 1)[-spatial_dims:]
            self.pool_modules.append(pool_type(kernel_size=size, stride=size))
            self.project_modules.append(
                conv_type(in_ch, 1, kernel_size=(1, 1, 1)[-spatial_dims:], stride=1, padding=(1, 1, 0)[-spatial_dims:])
            )

        self.spatial_dims = spatial_dims
        self.psp_block_num = psp_block_num
        self.upsample_mode = upsample_mode

        if self.upsample_mode == "transpose":
            conv_trans_type = Conv[Conv.CONVTRANS, spatial_dims]
            for i in range(psp_block_num):
                size = (2 ** (i + 3), 2 ** (i + 3), 1)[-spatial_dims:]
                pad_size = (2 ** (i + 3), 2 ** (i + 3), 0)[-spatial_dims:]
                self.up_modules.append(conv_trans_type(1, 1, kernel_size=size, stride=size, padding=pad_size))

    def forward(self, x):
        outputs = []
        if self.upsample_mode == "transpose":
            for (project_module, pool_module, up_module) in zip(
                self.project_modules, self.pool_modules, self.up_modules
            ):
                output = up_module(project_module(pool_module(x)))
                outputs.append(output)
        else:
            for (project_module, pool_module) in zip(self.project_modules, self.pool_modules):
                interpolate_size = x.shape[2:]
                align_corners: Optional[bool] = None
                if self.upsample_mode in ["trilinear", "bilinear"]:
                    align_corners = True
                output = F.interpolate(
                    project_module(pool_module(x)),
                    size=interpolate_size,
                    mode=self.upsample_mode,
                    align_corners=align_corners,
                )
                outputs.append(output)
        x = torch.cat(outputs, dim=1)
        return x


class AHNet(nn.Module):
    """
    AHNet based on `Anisotropic Hybrid Network <https://arxiv.org/pdf/1711.08580.pdf>`_.
    Adapted from `lsqshr's official code <https://github.com/lsqshr/AH-Net/blob/master/net3d.py>`_.
    Except from the original network that supports 3D inputs, this implementation also supports 2D inputs.
    According to the `tests for deconvolutions <https://github.com/Project-MONAI/MONAI/issues/1023>`_, using
    ``"transpose"`` rather than linear interpolations is faster. Therefore, this implementation sets ``"transpose"``
    as the default upsampling method.

    To meet the requirements of the structure, the input size for each spatial dimension
    (except the last one) should be: divisible by 2 ** (psp_block_num + 3) and no less than 32 in ``transpose`` mode,
    and should be divisible by 32 and no less than 2 ** (psp_block_num + 3) in other upsample modes.
    In addition, the input size for the last spatial dimension should be divisible by 32, and at least one spatial size
    should be no less than 64.

    Args:
        layers: number of residual blocks for 4 layers of the network (layer1...layer4). Defaults to ``(3, 4, 6, 3)``.
        spatial_dims: spatial dimension of the input data. Defaults to 3.
        in_channels: number of input channels for the network. Default to 1.
        out_channels: number of output channels for the network. Defaults to 1.
        psp_block_num: the number of pyramid volumetric pooling modules used at the end of the network before the final
            output layer for extracting multiscale features. The number should be an integer that belongs to [0,4]. Defaults
            to 4.
        upsample_mode: [``"transpose"``, ``"bilinear"``, ``"trilinear"``, ``nearest``]
            The mode of upsampling manipulations.
            Using the last two modes cannot guarantee the model's reproducibility. Defaults to ``transpose``.

            - ``"transpose"``, uses transposed convolution layers.
            - ``"bilinear"``, uses bilinear interpolate.
            - ``"trilinear"``, uses trilinear interpolate.
            - ``"nearest"``, uses nearest interpolate.
        pretrained: whether to load pretrained weights from ResNet50 to initialize convolution layers, default to False.
        progress: If True, displays a progress bar of the download of pretrained weights to stderr.
    """

    def __init__(
        self,
        layers: tuple = (3, 4, 6, 3),
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        psp_block_num: int = 4,
        upsample_mode: str = "transpose",
        pretrained: bool = False,
        progress: bool = True,
    ):
        self.inplanes = 64
        super().__init__()

        conv_type = Conv[Conv.CONV, spatial_dims]
        conv_trans_type = Conv[Conv.CONVTRANS, spatial_dims]
        norm_type = Norm[Norm.BATCH, spatial_dims]
        pool_type: Type[Union[nn.MaxPool2d, nn.MaxPool3d]] = Pool[Pool.MAX, spatial_dims]
        relu_type: Type[nn.ReLU] = Act[Act.RELU]
        conv2d_type: Type[nn.Conv2d] = Conv[Conv.CONV, 2]
        norm2d_type: Type[nn.BatchNorm2d] = Norm[Norm.BATCH, 2]

        self.conv2d_type = conv2d_type
        self.norm2d_type = norm2d_type
        self.conv_type = conv_type
        self.norm_type = norm_type
        self.relu_type = relu_type
        self.pool_type = pool_type
        self.spatial_dims = spatial_dims
        self.psp_block_num = psp_block_num
        self.psp = None

        if spatial_dims not in [2, 3]:
            raise AssertionError("spatial_dims can only be 2 or 3.")
        if psp_block_num not in [0, 1, 2, 3, 4]:
            raise AssertionError("psp_block_num should be an integer that belongs to [0, 4].")

        self.conv1 = conv_type(
            in_channels,
            64,
            kernel_size=(7, 7, 3)[-spatial_dims:],
            stride=(2, 2, 1)[-spatial_dims:],
            padding=(3, 3, 1)[-spatial_dims:],
            bias=False,
        )
        self.pool1 = pool_type(kernel_size=(1, 1, 2)[-spatial_dims:], stride=(1, 1, 2)[-spatial_dims:])
        self.bn0 = norm_type(64)
        self.relu = relu_type(inplace=True)
        if upsample_mode in ["transpose", "nearest"]:
            # To maintain the determinism, the value of kernel_size and stride should be the same.
            # (you can check this link for reference: https://github.com/Project-MONAI/MONAI/pull/815 )
            self.maxpool = pool_type(kernel_size=(2, 2, 2)[-spatial_dims:], stride=2)
        else:
            self.maxpool = pool_type(kernel_size=(3, 3, 3)[-spatial_dims:], stride=2, padding=1)

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

        self.up0 = UpTransition(spatial_dims, noutres4, noutres3, upsample_mode)
        self.dense0 = DenseBlock(spatial_dims, ndenselayer, noutres3, densebn, densegrowth, 0.0)
        noutdense = noutres3 + ndenselayer * densegrowth

        self.up1 = UpTransition(spatial_dims, noutdense, noutres2, upsample_mode)
        self.dense1 = DenseBlock(spatial_dims, ndenselayer, noutres2, densebn, densegrowth, 0.0)
        noutdense1 = noutres2 + ndenselayer * densegrowth

        self.up2 = UpTransition(spatial_dims, noutdense1, noutres1, upsample_mode)
        self.dense2 = DenseBlock(spatial_dims, ndenselayer, noutres1, densebn, densegrowth, 0.0)
        noutdense2 = noutres1 + ndenselayer * densegrowth

        self.trans1 = Projection(spatial_dims, noutdense2, num_init_features)
        self.dense3 = DenseBlock(spatial_dims, ndenselayer, num_init_features, densebn, densegrowth, 0.0)
        noutdense3 = num_init_features + densegrowth * ndenselayer

        self.up3 = UpTransition(spatial_dims, noutdense3, num_init_features, upsample_mode)
        self.dense4 = DenseBlock(spatial_dims, ndenselayer, num_init_features, densebn, densegrowth, 0.0)
        noutdense4 = num_init_features + densegrowth * ndenselayer

        self.psp = PSP(spatial_dims, psp_block_num, noutdense4, upsample_mode)
        self.final = Final(spatial_dims, psp_block_num + noutdense4, out_channels, upsample_mode)

        # Initialise parameters
        for m in self.modules():
            if isinstance(m, (conv_type, conv_trans_type)):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, norm_type):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if pretrained:
            net2d = FCN(pretrained=True, progress=progress)
            self.copy_from(net2d)

    def _make_layer(self, block: Type[Bottleneck3x3x1], planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.conv_type(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(stride, stride, 1)[: self.spatial_dims],
                    bias=False,
                ),
                self.pool_type(
                    kernel_size=(1, 1, stride)[: self.spatial_dims], stride=(1, 1, stride)[: self.spatial_dims]
                ),
                self.norm_type(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.spatial_dims, self.inplanes, planes, (stride, stride, 1)[: self.spatial_dims], downsample)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.spatial_dims, self.inplanes, planes))
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
        if self.psp_block_num > 0:
            psp = self.psp(d4)
            x = torch.cat((psp, d4), dim=1)
        else:
            x = d4
        return self.final(x)

    def copy_from(self, net):
        # This method only supports for 3D AHNet, the input channel should be 1.
        p2d, p3d = next(net.conv1.parameters()), next(self.conv1.parameters())

        # From 64x3x7x7 -> 64x3x7x7x1 -> 64x1x7x7x3
        weights = p2d.data.unsqueeze(dim=4).permute(0, 4, 2, 3, 1).clone()
        p3d.data = weights.repeat([1, p3d.shape[1], 1, 1, 1])

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
                if isinstance(m2, (self.norm_type, self.conv_type)):
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


AHnet = Ahnet = AHNet
