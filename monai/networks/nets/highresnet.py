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

import torch.nn as nn
import torch.nn.functional as F

from monai.networks.layers.convutils import same_padding
from monai.networks.layers.factories import (get_conv_type, get_dropout_type, get_normalize_type)

SUPPORTED_NORM = {
    'batch': lambda spatial_dims: get_normalize_type(spatial_dims, is_instance=False),
    'instance': lambda spatial_dims: get_normalize_type(spatial_dims, is_instance=True),
}
SUPPORTED_ACTI = {'relu': nn.ReLU, 'prelu': nn.PReLU, 'relu6': nn.ReLU6}
DEFAULT_LAYER_PARAMS_3D = (
    # initial conv layer
    {'name': 'conv_0', 'n_features': 16, 'kernel_size': 3},
    # residual blocks
    {'name': 'res_1', 'n_features': 16, 'kernels': (3, 3), 'repeat': 3},
    {'name': 'res_2', 'n_features': 32, 'kernels': (3, 3), 'repeat': 3},
    {'name': 'res_3', 'n_features': 64, 'kernels': (3, 3), 'repeat': 3},
    # final conv layers
    {'name': 'conv_1', 'n_features': 80, 'kernel_size': 1},
    {'name': 'conv_2', 'kernel_size': 1},
)


class ConvNormActi(nn.Module):

    def __init__(self,
                 spatial_dims,
                 in_channels,
                 out_channels,
                 kernel_size,
                 norm_type=None,
                 acti_type=None,
                 dropout_prob=None):

        super(ConvNormActi, self).__init__()

        layers = nn.ModuleList()

        conv_type = get_conv_type(spatial_dims, is_transpose=False)
        padding_size = same_padding(kernel_size)
        conv = conv_type(in_channels, out_channels, kernel_size, padding=padding_size)
        layers.append(conv)

        if norm_type is not None:
            layers.append(SUPPORTED_NORM[norm_type](spatial_dims)(out_channels))
        if acti_type is not None:
            layers.append(SUPPORTED_ACTI[acti_type](inplace=True))
        if dropout_prob is not None:
            dropout_type = get_dropout_type(spatial_dims)
            layers.append(dropout_type(p=dropout_prob))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class HighResBlock(nn.Module):

    def __init__(self,
                 spatial_dims,
                 in_channels,
                 out_channels,
                 kernels=(3, 3),
                 dilation=1,
                 norm_type='instance',
                 acti_type='relu',
                 channel_matching='pad'):
        """
        Args:
            kernels (list of int): each integer k in `kernels` corresponds to a convolution layer with kernel size k.
            channel_matching ('pad'|'project'): handling residual branch and conv branch channel mismatches
                with either zero padding ('pad') or a trainable conv with kernel size 1 ('project').
        """
        super(HighResBlock, self).__init__()
        conv_type = get_conv_type(spatial_dims, is_transpose=False)

        self.project, self.pad = None, None
        if in_channels != out_channels:
            if channel_matching not in ('pad', 'project'):
                raise ValueError('channel matching must be pad or project, got {}.'.format(channel_matching))
            if channel_matching == 'project':
                self.project = conv_type(in_channels, out_channels, kernel_size=1)
            if channel_matching == 'pad':
                if in_channels > out_channels:
                    raise ValueError('in_channels > out_channels is incompatible with `channel_matching=pad`.')
                pad_1 = (out_channels - in_channels) // 2
                pad_2 = out_channels - in_channels - pad_1
                pad = [0, 0] * spatial_dims + [pad_1, pad_2] + [0, 0]
                self.pad = lambda input: F.pad(input, pad)

        layers = nn.ModuleList()
        _in_chns, _out_chns = in_channels, out_channels
        for kernel_size in kernels:
            layers.append(SUPPORTED_NORM[norm_type](spatial_dims)(_in_chns))
            layers.append(SUPPORTED_ACTI[acti_type](inplace=True))
            layers.append(
                conv_type(_in_chns,
                          _out_chns,
                          kernel_size,
                          padding=same_padding(kernel_size, dilation),
                          dilation=dilation))
            _in_chns = _out_chns
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x_conv = self.layers(x)
        if self.project is not None:
            return x_conv + self.project(x)
        if self.pad is not None:
            return x_conv + self.pad(x)
        return x_conv + x


class HighResNet(nn.Module):
    """
    Reimplementation of highres3dnet based on
    Li et al., "On the compactness, efficiency, and representation of 3D
    convolutional networks: Brain parcellation as a pretext task", IPMI '17

    Adapted from:
    https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/network/highres3dnet.py
    https://github.com/fepegar/highresnet

    Args:
        spatial_dims (int): number of spatial dimensions of the input image.
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        norm_type ('batch'|'instance'): feature normalisation with batchnorm or instancenorm.
        acti_type ('relu'|'prelu'|'relu6'): non-linear activation using ReLU or PReLU.
        dropout_prob (float): probability of the feature map to be zeroed
            (only applies to the penultimate conv layer).
        layer_params (a list of dictionaries): specifying key paraemters of each layer/block.
    """

    def __init__(self,
                 spatial_dims=3,
                 in_channels=1,
                 out_channels=1,
                 norm_type='batch',
                 acti_type='relu',
                 dropout_prob=None,
                 layer_params=DEFAULT_LAYER_PARAMS_3D):

        super(HighResNet, self).__init__()
        blocks = nn.ModuleList()

        # intial conv layer
        params = layer_params[0]
        _in_chns, _out_chns = in_channels, params['n_features']
        blocks.append(
            ConvNormActi(spatial_dims,
                         _in_chns,
                         _out_chns,
                         kernel_size=params['kernel_size'],
                         norm_type=norm_type,
                         acti_type=acti_type,
                         dropout_prob=None))

        # residual blocks
        for (idx, params) in enumerate(layer_params[1:-2]):  # res blocks except the 1st and last two conv layers.
            _in_chns, _out_chns = _out_chns, params['n_features']
            _dilation = 2**idx
            for _ in range(params['repeat']):
                blocks.append(
                    HighResBlock(spatial_dims,
                                 _in_chns,
                                 _out_chns,
                                 params['kernels'],
                                 dilation=_dilation,
                                 norm_type=norm_type,
                                 acti_type=acti_type))
                _in_chns = _out_chns

        # final conv layers
        params = layer_params[-2]
        _in_chns, _out_chns = _out_chns, params['n_features']
        blocks.append(
            ConvNormActi(spatial_dims,
                         _in_chns,
                         _out_chns,
                         kernel_size=params['kernel_size'],
                         norm_type=norm_type,
                         acti_type=acti_type,
                         dropout_prob=dropout_prob))

        params = layer_params[-1]
        _in_chns = _out_chns
        blocks.append(
            ConvNormActi(spatial_dims,
                         _in_chns,
                         out_channels,
                         kernel_size=params['kernel_size'],
                         norm_type=norm_type,
                         acti_type=None,
                         dropout_prob=None))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)
