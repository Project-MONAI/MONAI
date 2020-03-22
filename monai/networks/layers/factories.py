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

"""
handles spatial 1D, 2D, 3D network components with a factory pattern.
"""

from torch import nn as nn


def get_conv_type(dim, is_transpose):
    if is_transpose:
        types = [nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
    else:
        types = [nn.Conv1d, nn.Conv2d, nn.Conv3d]

    return types[dim - 1]


def get_dropout_type(dim):
    types = [nn.Dropout, nn.Dropout2d, nn.Dropout3d]
    return types[dim - 1]


def get_normalize_type(dim, is_instance):
    if is_instance:
        types = [nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d]
    else:
        types = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]

    return types[dim - 1]


def get_maxpooling_type(dim, is_adaptive):
    if is_adaptive:
        types = [nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d]
    else:
        types = [nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d]
    return types[dim - 1]


def get_avgpooling_type(dim, is_adaptive):
    if is_adaptive:
        types = [nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]
    else:
        types = [nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d]
    return types[dim - 1]
