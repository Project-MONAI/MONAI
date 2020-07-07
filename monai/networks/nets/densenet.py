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

from collections import OrderedDict
from typing import Callable, Any

import torch
import torch.nn as nn

from monai.networks.layers.factories import Conv, Dropout, Pool, Norm


class _DenseLayer(nn.Sequential):
    def __init__(self, spatial_dims, in_channels, growth_rate, bn_size, dropout_prob) -> None:
        super(_DenseLayer, self).__init__()

        out_channels = bn_size * growth_rate
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]
        dropout_type: Callable = Dropout[Dropout.DROPOUT, spatial_dims]

        self.add_module("norm1", norm_type(in_channels))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", conv_type(in_channels, out_channels, kernel_size=1, bias=False))

        self.add_module("norm2", norm_type(out_channels))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", conv_type(out_channels, growth_rate, kernel_size=3, padding=1, bias=False))

        if dropout_prob > 0:
            self.add_module("dropout", dropout_type(dropout_prob))

    def forward(self, x) -> Any:
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, spatial_dims, layers: int, in_channels, bn_size, growth_rate, dropout_prob) -> None:
        super(_DenseBlock, self).__init__()
        for i in range(layers):
            layer = _DenseLayer(spatial_dims, in_channels, growth_rate, bn_size, dropout_prob)
            in_channels += growth_rate
            self.add_module("denselayer%d" % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, spatial_dims, in_channels, out_channels) -> None:
        super(_Transition, self).__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]
        pool_type: Callable = Pool[Pool.AVG, spatial_dims]

        self.add_module("norm", norm_type(in_channels))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", conv_type(in_channels, out_channels, kernel_size=1, bias=False))
        self.add_module("pool", pool_type(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """
    Densenet based on: "Densely Connected Convolutional Networks" https://arxiv.org/pdf/1608.06993.pdf
    Adapted from PyTorch Hub 2D version:
    https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

    Args:
        spatial_dims: number of spatial dimensions of the input image.
        in_channels: number of the input channel.
        out_channels: number of the output classes.
        init_features: number of filters in the first convolution layer.
        growth_rate: how many filters to add each layer (k in paper).
        block_config (tuple): how many layers in each pooling block.
        bn_size: multiplicative factor for number of bottle neck layers.
                      (i.e. bn_size * k features in the bottleneck layer)
        dropout_prob: dropout rate after each dense layer.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config=(6, 12, 24, 16),
        bn_size: int = 4,
        dropout_prob: float = 0.0,
    ):

        super(DenseNet, self).__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]
        pool_type: Callable = Pool[Pool.MAX, spatial_dims]
        avg_pool_type: Callable = Pool[Pool.ADAPTIVEAVG, spatial_dims]

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", conv_type(in_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", norm_type(init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", pool_type(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        in_channels = init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                spatial_dims=spatial_dims,
                layers=num_layers,
                in_channels=in_channels,
                bn_size=bn_size,
                growth_rate=growth_rate,
                dropout_prob=dropout_prob,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            in_channels += num_layers * growth_rate
            if i == len(block_config) - 1:
                self.features.add_module("norm5", norm_type(in_channels))
            else:
                _out_channels = in_channels // 2
                trans = _Transition(spatial_dims, in_channels=in_channels, out_channels=_out_channels)
                self.features.add_module("transition%d" % (i + 1), trans)
                in_channels = _out_channels

        # pooling and classification
        self.class_layers = nn.Sequential(
            OrderedDict(
                [
                    ("relu", nn.ReLU(inplace=True)),
                    ("norm", avg_pool_type(1)),
                    ("flatten", nn.Flatten(1)),
                    ("class", nn.Linear(in_channels, out_channels)),
                ]
            )
        )

        # Avoid Built-in function isinstance was called with the wrong arguments warning
        # pytype: disable=wrong-arg-types
        for m in self.modules():
            if isinstance(m, conv_type):  # type: ignore
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, norm_type):  # type: ignore
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):  # type: ignore
                nn.init.constant_(m.bias, 0)
        # pytype: enable=wrong-arg-types

    def forward(self, x):
        x = self.features(x)
        x = self.class_layers(x)
        return x


def densenet121(**kwargs) -> DenseNet:
    model = DenseNet(init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs)
    return model


def densenet169(**kwargs) -> DenseNet:
    model = DenseNet(init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), **kwargs)
    return model


def densenet201(**kwargs) -> DenseNet:
    model = DenseNet(init_features=64, growth_rate=32, block_config=(6, 12, 48, 32), **kwargs)
    return model


def densenet264(**kwargs) -> DenseNet:
    model = DenseNet(init_features=64, growth_rate=32, block_config=(6, 12, 64, 48), **kwargs)
    return model
