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

import re
from collections import OrderedDict
from typing import Any, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.squeeze_and_excitation import SEBottleneck, SEResNetBottleneck, SEResNeXtBottleneck
from monai.networks.layers.factories import Act, Conv, Dropout, Norm, Pool


class SENet(nn.Module):
    """
    SENet based on `Squeeze-and-Excitation Networks <https://arxiv.org/pdf/1709.01507.pdf>`_.
    Adapted from `Cadene Hub 2D version
    <https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py>`_.

    Args:
        spatial_dims: spatial dimension of the input data.
        in_channels: channel number of the input data.
        block: SEBlock class.
            for SENet154: SEBottleneck
            for SE-ResNet models: SEResNetBottleneck
            for SE-ResNeXt models:  SEResNeXtBottleneck
        layers: number of residual blocks for 4 layers of the network (layer1...layer4).
        groups: number of groups for the 3x3 convolution in each bottleneck block.
            for SENet154: 64
            for SE-ResNet models: 1
            for SE-ResNeXt models:  32
        reduction: reduction ratio for Squeeze-and-Excitation modules.
            for all models: 16
        dropout_prob: drop probability for the Dropout layer.
            if `None` the Dropout layer is not used.
            for SENet154: 0.2
            for SE-ResNet models: None
            for SE-ResNeXt models: None
        dropout_dim: determine the dimensions of dropout. Defaults to 1.
            When dropout_dim = 1, randomly zeroes some of the elements for each channel.
            When dropout_dim = 2, Randomly zeroes out entire channels (a channel is a 2D feature map).
            When dropout_dim = 3, Randomly zeroes out entire channels (a channel is a 3D feature map).
        inplanes:  number of input channels for layer1.
            for SENet154: 128
            for SE-ResNet models: 64
            for SE-ResNeXt models: 64
        downsample_kernel_size: kernel size for downsampling convolutions in layer2, layer3 and layer4.
            for SENet154: 3
            for SE-ResNet models: 1
            for SE-ResNeXt models: 1
        input_3x3: If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        num_classes: number of outputs in `last_linear` layer.
            for all models: 1000

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        block: Type[Union[SEBottleneck, SEResNetBottleneck, SEResNeXtBottleneck]],
        layers: List[int],
        groups: int,
        reduction: int,
        dropout_prob: Optional[float] = 0.2,
        dropout_dim: int = 1,
        inplanes: int = 128,
        downsample_kernel_size: int = 3,
        input_3x3: bool = True,
        num_classes: int = 1000,
    ) -> None:

        super(SENet, self).__init__()

        relu_type: Type[nn.ReLU] = Act[Act.RELU]
        conv_type: Type[Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]] = Conv[Conv.CONV, spatial_dims]
        pool_type: Type[Union[nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d]] = Pool[Pool.MAX, spatial_dims]
        norm_type: Type[Union[nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]] = Norm[Norm.BATCH, spatial_dims]
        dropout_type: Type[Union[nn.Dropout, nn.Dropout2d, nn.Dropout3d]] = Dropout[Dropout.DROPOUT, dropout_dim]
        avg_pool_type: Type[Union[nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]] = Pool[
            Pool.ADAPTIVEAVG, spatial_dims
        ]

        self.inplanes = inplanes
        self.spatial_dims = spatial_dims

        layer0_modules: List[Tuple[str, Any]]

        if input_3x3:
            layer0_modules = [
                (
                    "conv1",
                    conv_type(in_channels=in_channels, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
                ),
                ("bn1", norm_type(num_features=64)),
                ("relu1", relu_type(inplace=True)),
                ("conv2", conv_type(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)),
                ("bn2", norm_type(num_features=64)),
                ("relu2", relu_type(inplace=True)),
                (
                    "conv3",
                    conv_type(in_channels=64, out_channels=inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                ),
                ("bn3", norm_type(num_features=inplanes)),
                ("relu3", relu_type(inplace=True)),
            ]
        else:
            layer0_modules = [
                (
                    "conv1",
                    conv_type(
                        in_channels=in_channels, out_channels=inplanes, kernel_size=7, stride=2, padding=3, bias=False
                    ),
                ),
                ("bn1", norm_type(num_features=inplanes)),
                ("relu1", relu_type(inplace=True)),
            ]

        layer0_modules.append(("pool", pool_type(kernel_size=3, stride=2, ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block, planes=64, blocks=layers[0], groups=groups, reduction=reduction, downsample_kernel_size=1
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
        )
        self.adaptive_avg_pool = avg_pool_type(1)
        self.dropout = dropout_type(dropout_prob) if dropout_prob is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight))
            elif isinstance(m, norm_type):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)

    def _make_layer(
        self,
        block: Type[Union[SEBottleneck, SEResNetBottleneck, SEResNeXtBottleneck]],
        planes: int,
        blocks: int,
        groups: int,
        reduction: int,
        stride: int = 1,
        downsample_kernel_size: int = 1,
    ) -> nn.Sequential:

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Convolution(
                dimensions=self.spatial_dims,
                in_channels=self.inplanes,
                out_channels=planes * block.expansion,
                strides=stride,
                kernel_size=downsample_kernel_size,
                act=None,
                norm=Norm.BATCH,
                bias=False,
            )

        layers = []
        layers.append(
            block(
                spatial_dims=self.spatial_dims,
                inplanes=self.inplanes,
                planes=planes,
                groups=groups,
                reduction=reduction,
                stride=stride,
                downsample=downsample,
            )
        )
        self.inplanes = planes * block.expansion
        for _num in range(1, blocks):
            layers.append(
                block(
                    spatial_dims=self.spatial_dims,
                    inplanes=self.inplanes,
                    planes=planes,
                    groups=groups,
                    reduction=reduction,
                )
            )

        return nn.Sequential(*layers)

    def features(self, x: torch.Tensor):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x: torch.Tensor):
        x = self.adaptive_avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.last_linear(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.logits(x)
        return x


model_urls = {
    "senet154": "http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth",
    "se_resnet50": "http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth",
    "se_resnet101": "http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth",
    "se_resnet152": "http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth",
    "se_resnext50_32x4d": "http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth",
    "se_resnext101_32x4d": "http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth",
}


def _load_state_dict(model, model_url, progress):
    """
    This function is used to load pretrained models.
    """
    pattern_conv = re.compile(r"^(layer[1-4]\.\d\.(?:conv)\d\.)(\w*)$")
    pattern_bn = re.compile(r"^(layer[1-4]\.\d\.)(?:bn)(\d\.)(\w*)$")
    pattern_se = re.compile(r"^(layer[1-4]\.\d\.)(?:se_module.fc1.)(\w*)$")
    pattern_se2 = re.compile(r"^(layer[1-4]\.\d\.)(?:se_module.fc2.)(\w*)$")
    pattern_down_conv = re.compile(r"^(layer[1-4]\.\d\.)(?:downsample.0.)(\w*)$")
    pattern_down_bn = re.compile(r"^(layer[1-4]\.\d\.)(?:downsample.1.)(\w*)$")

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        new_key = None
        if pattern_conv.match(key):
            new_key = re.sub(pattern_conv, r"\1conv.\2", key)
        elif pattern_bn.match(key):
            new_key = re.sub(pattern_bn, r"\1conv\2norm.\3", key)
        elif pattern_se.match(key):
            state_dict[key] = state_dict[key].squeeze()
            new_key = re.sub(pattern_se, r"\1se_layer.fc.0.\2", key)
        elif pattern_se2.match(key):
            state_dict[key] = state_dict[key].squeeze()
            new_key = re.sub(pattern_se2, r"\1se_layer.fc.2.\2", key)
        elif pattern_down_conv.match(key):
            new_key = re.sub(pattern_down_conv, r"\1project.conv.\2", key)
        elif pattern_down_bn.match(key):
            new_key = re.sub(pattern_down_bn, r"\1project.norm.\2", key)
        if new_key:
            state_dict[new_key] = state_dict[key]
            del state_dict[key]

    model_dict = model.state_dict()
    state_dict = {
        k: v for k, v in state_dict.items() if (k in model_dict) and (model_dict[k].shape == state_dict[k].shape)
    }
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)


def senet154(
    spatial_dims: int,
    in_channels: int,
    num_classes: int,
    pretrained: bool = False,
    progress: bool = True,
) -> SENet:
    """
    when `spatial_dims = 2`, specify `pretrained = True` can load Imagenet pretrained weights achieved
    from `Cadene Hub 2D version
    <https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py>`_.
    """
    model = SENet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        block=SEBottleneck,
        layers=[3, 8, 36, 3],
        groups=64,
        reduction=16,
        dropout_prob=0.2,
        dropout_dim=1,
        num_classes=num_classes,
    )
    if pretrained:
        arch = "senet154"
        _load_state_dict(model, model_urls[arch], progress)
    return model


def se_resnet50(
    spatial_dims: int, in_channels: int, num_classes: int, pretrained: bool = False, progress: bool = True
) -> SENet:
    """
    when `spatial_dims = 2`, specify `pretrained = True` can load Imagenet pretrained weights achieved
    from `Cadene Hub 2D version
    <https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py>`_.
    """
    model = SENet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        block=SEResNetBottleneck,
        layers=[3, 4, 6, 3],
        groups=1,
        reduction=16,
        dropout_prob=None,
        inplanes=64,
        input_3x3=False,
        downsample_kernel_size=1,
        num_classes=num_classes,
    )
    if pretrained:
        arch = "se_resnet50"
        _load_state_dict(model, model_urls[arch], progress)
    return model


def se_resnet101(
    spatial_dims: int, in_channels: int, num_classes: int, pretrained: bool = False, progress: bool = True
) -> SENet:
    """
    when `spatial_dims = 2`, specify `pretrained = True` can load Imagenet pretrained weights achieved
    from `Cadene Hub 2D version
    <https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py>`_.
    """
    model = SENet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        block=SEResNetBottleneck,
        layers=[3, 4, 23, 3],
        groups=1,
        reduction=16,
        dropout_prob=0.2,
        dropout_dim=1,
        inplanes=64,
        input_3x3=False,
        downsample_kernel_size=1,
        num_classes=num_classes,
    )
    if pretrained:
        arch = "se_resnet101"
        _load_state_dict(model, model_urls[arch], progress)
    return model


def se_resnet152(
    spatial_dims: int, in_channels: int, num_classes: int, pretrained: bool = False, progress: bool = True
) -> SENet:
    """
    when `spatial_dims = 2`, specify `pretrained = True` can load Imagenet pretrained weights achieved
    from `Cadene Hub 2D version
    <https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py>`_.
    """
    model = SENet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        block=SEResNetBottleneck,
        layers=[3, 8, 36, 3],
        groups=1,
        reduction=16,
        dropout_prob=0.2,
        dropout_dim=1,
        inplanes=64,
        input_3x3=False,
        downsample_kernel_size=1,
        num_classes=num_classes,
    )
    if pretrained:
        arch = "se_resnet152"
        _load_state_dict(model, model_urls[arch], progress)
    return model


def se_resnext50_32x4d(
    spatial_dims: int, in_channels: int, num_classes: int, pretrained: bool = False, progress: bool = True
) -> SENet:
    """
    when `spatial_dims = 2`, specify `pretrained = True` can load Imagenet pretrained weights achieved
    from `Cadene Hub 2D version
    <https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py>`_.
    """
    model = SENet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        block=SEResNeXtBottleneck,
        layers=[3, 4, 6, 3],
        groups=32,
        reduction=16,
        dropout_prob=None,
        inplanes=64,
        input_3x3=False,
        downsample_kernel_size=1,
        num_classes=num_classes,
    )
    if pretrained:
        arch = "se_resnext50_32x4d"
        _load_state_dict(model, model_urls[arch], progress)
    return model


def se_resnext101_32x4d(
    spatial_dims: int, in_channels: int, num_classes: int, pretrained: bool = False, progress: bool = True
) -> SENet:
    """
    when `spatial_dims = 2`, specify `pretrained = True` can load Imagenet pretrained weights achieved
    from `Cadene Hub 2D version
    <https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py>`_.
    """
    model = SENet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        block=SEResNeXtBottleneck,
        layers=[3, 4, 23, 3],
        groups=32,
        reduction=16,
        dropout_prob=None,
        inplanes=64,
        input_3x3=False,
        downsample_kernel_size=1,
        num_classes=num_classes,
    )
    if pretrained:
        arch = "se_resnext101_32x4d"
        _load_state_dict(model, model_urls[arch], progress)
    return model
