# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =========================================================================
# Adapted from https://github.com/pytorch/vision/blob/release/0.12/torchvision/ops/feature_pyramid_network.py
# which has the following license...
# https://github.com/pytorch/vision/blob/main/LICENSE
#
# BSD 3-Clause License

# Copyright (c) Soumith Chintala 2016,
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
This script is modified from from torchvision to support N-D images,
by overriding the definition of convolutional layers and pooling layers.

https://github.com/pytorch/vision/blob/release/0.12/torchvision/ops/feature_pyramid_network.py
"""

from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch.nn.functional as F
from torch import Tensor, nn

from monai.networks.layers.factories import Conv, Pool

__all__ = ["ExtraFPNBlock", "LastLevelMaxPool", "LastLevelP6P7", "FeaturePyramidNetwork"]


class ExtraFPNBlock(nn.Module):
    """
    Base class for the extra block in the FPN.

    Same code as https://github.com/pytorch/vision/blob/release/0.12/torchvision/ops/feature_pyramid_network.py
    """

    def forward(self, results: List[Tensor], x: List[Tensor], names: List[str]) -> Tuple[List[Tensor], List[str]]:
        """
        Compute extended set of results of the FPN and their names.

        Args:
            results: the result of the FPN
            x: the original feature maps
            names: the names for each one of the original feature maps

        Returns:
            - the extended set of results of the FPN
            - the extended set of names for the results
        """
        pass


class LastLevelMaxPool(ExtraFPNBlock):
    """
    Applies a max_pool2d or max_pool3d on top of the last feature map. Serves as an ``extra_blocks``
    in :class:`~monai.apps.detection.networks.blocks.feature_pyramid_network.FeaturePyramidNetwork` .
    """

    def forward(self, results: List[Tensor], x: List[Tensor], names: List[str]) -> Tuple[List[Tensor], List[str]]:
        spatial_dims = len(results[0].shape) - 2
        pool_type: Type[Union[nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d]] = Pool[Pool.MAX, spatial_dims]
        self.maxpool = pool_type(kernel_size=1, stride=2, padding=0)

        names.append("pool")
        results.append(self.maxpool(results[-1]))
        return results, names


class LastLevelP6P7(ExtraFPNBlock):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    Serves as an ``extra_blocks``
    in :class:`~monai.apps.detection.networks.blocks.feature_pyramid_network.FeaturePyramidNetwork` .
    """

    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int):
        super().__init__()
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        self.p6 = conv_type(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.p7 = conv_type(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, results: List[Tensor], x: List[Tensor], names: List[str]) -> Tuple[List[Tensor], List[str]]:
        p5, c5 = results[-1], x[-1]
        x5 = p5 if self.use_P5 else c5
        p6 = self.p6(x5)
        p7 = self.p7(F.relu(p6))
        results.extend([p6, p7])
        names.extend(["p6", "p7"])
        return results, names


class FeaturePyramidNetwork(nn.Module):
    """
    Module that adds a FPN from on top of a set of feature maps. This is based on
    `"Feature Pyramid Network for Object Detection" <https://arxiv.org/abs/1612.03144>`_.

    The feature maps are currently supposed to be in increasing depth
    order.

    The input to the model is expected to be an OrderedDict[Tensor], containing
    the feature maps on top of which the FPN will be added.

    Args:
        spatial_dims: 2D or 3D images
        in_channels_list: number of channels for each feature map that
            is passed to the module
        out_channels: number of channels of the FPN representation
        extra_blocks: if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names

    Examples::

        >>> m = FeaturePyramidNetwork(2, [10, 20, 30], 5)
        >>> # get some dummy data
        >>> x = OrderedDict()
        >>> x['feat0'] = torch.rand(1, 10, 64, 64)
        >>> x['feat2'] = torch.rand(1, 20, 16, 16)
        >>> x['feat3'] = torch.rand(1, 30, 8, 8)
        >>> # compute the FPN on top of x
        >>> output = m(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        >>> # returns
        >>>   [('feat0', torch.Size([1, 5, 64, 64])),
        >>>    ('feat2', torch.Size([1, 5, 16, 16])),
        >>>    ('feat3', torch.Size([1, 5, 8, 8]))]

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
    ):
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]

        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")
            inner_block_module = conv_type(in_channels, out_channels, 1)
            layer_block_module = conv_type(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        conv_type_: Type[nn.Module] = Conv[Conv.CONV, spatial_dims]
        for m in self.modules():
            if isinstance(m, conv_type_):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0.0)  # type: ignore

        if extra_blocks is not None:
            assert isinstance(extra_blocks, ExtraFPNBlock)
        self.extra_blocks = extra_blocks

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.inner_blocks):
            if i == idx:
                out = module(x)
        return out

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.layer_blocks):
            if i == idx:
                out = module(x)
        return out

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Computes the FPN for a set of feature maps.

        Args:
            x: feature maps for each feature level.

        Returns:
            feature maps after FPN layers. They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x_values: List = list(x.values())

        last_inner = self.get_result_from_inner_blocks(x_values[-1], -1)
        results = []
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        for idx in range(len(x_values) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x_values[idx], idx)
            feat_shape = inner_lateral.shape[2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x_values, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out
