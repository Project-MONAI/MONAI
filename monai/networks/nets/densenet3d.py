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

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def densenet121(**kwargs):
    model = DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        **kwargs)
    return model


def densenet169(**kwargs):
    model = DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 32, 32),
        **kwargs)
    return model


def densenet201(**kwargs):
    model = DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 48, 32),
        **kwargs)
    return model


def densenet264(**kwargs):
    model = DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 64, 48),
        **kwargs)
    return model


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('denseblock{}'.format(i))
        ft_module_names.append('transition{}'.format(i))
    ft_module_names.append('norm5')
    ft_module_names.append('classifier')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1',
                        nn.Conv3d(
                            num_input_features,
                            bn_size * growth_rate,
                            kernel_size=1,
                            stride=1,
                            bias=False))
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2',
                        nn.Conv3d(
                            bn_size * growth_rate,
                            growth_rate,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv',
                        nn.Conv3d(
                            num_input_features,
                            num_output_features,
                            kernel_size=1,
                            stride=1,
                            bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Densenet-BC model class.
    Based on: "Densely Connected Convolutional Networks" https://arxiv.org/pdf/1608.06993.pdf
    Adapted from PyTorch Hub 2D version:
    https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

    Args:
        num_input_channel (Int): number of the input image channel.
        num_classes (Int): number of classification classes.
        growth_rate (Int): how many filters to add each layer (k in paper).
        block_config (list): how many layers in each pooling block.
        num_init_features (Int) the number of filters to learn in the first convolution layer.
        bn_size (Int) multiplicative factor for number of bottle neck layers.
                      (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (Float): dropout rate after each dense layer.
    """

    def __init__(self,
                 num_input_channel=3,
                 num_classes=1000,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0):

        super(DenseNet, self).__init__()
        # First convolution
        self.features = nn.Sequential(
            OrderedDict([
                ('conv0',
                 nn.Conv3d(
                     num_input_channel,
                     num_init_features,
                     kernel_size=7,
                     stride=2,
                     padding=3,
                     bias=False)),
                ('norm0', nn.BatchNorm3d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
            ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
