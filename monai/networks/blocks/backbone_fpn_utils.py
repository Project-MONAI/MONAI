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
# Adapted from https://github.com/pytorch/vision/blob/release/0.12/torchvision/models/detection/backbone_utils.py
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

https://github.com/pytorch/vision/blob/release/0.12/torchvision/models/detection/backbone_utils.py
"""

from typing import Dict, List, Optional, Union

from torch import Tensor, nn

from monai.networks.nets import resnet
from monai.utils import optional_import

from .feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork, LastLevelMaxPool

torchvision_models, _ = optional_import("torchvision.models")

__all__ = ["BackboneWithFPN"]


class BackboneWithFPN(nn.Module):
    """
    Adds an FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediateLayerGetter apply here.

    Same code as https://github.com/pytorch/vision/blob/release/0.12/torchvision/models/detection/backbone_utils.py
    Except that this class uses spatial_dims

    Args:
        backbone: backbone network
        return_layers: a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list: number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels: number of channels in the FPN.
        spatial_dims: 2D or 3D images
    """

    def __init__(
        self,
        backbone: nn.Module,
        return_layers: Dict[str, str],
        in_channels_list: List[int],
        out_channels: int,
        spatial_dims: Union[int, None] = None,
        extra_blocks: Optional[ExtraFPNBlock] = None,
    ) -> None:
        super().__init__()

        # if spatial_dims is not specified, try to find it from backbone.
        if spatial_dims is None:
            if hasattr(backbone, "spatial_dims") and isinstance(backbone.spatial_dims, int):
                spatial_dims = backbone.spatial_dims
            elif isinstance(backbone.conv1, nn.Conv2d):
                spatial_dims = 2
            elif isinstance(backbone.conv1, nn.Conv3d):
                spatial_dims = 3
            else:
                raise ValueError("Could not find spatial_dims of backbone, please specify it.")

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool(spatial_dims)

        self.body = torchvision_models._utils.IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            spatial_dims=spatial_dims,
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Computes the resulted feature maps of the network.

        Args:
            x: input images

        Returns:
            feature maps after FPN layers. They are ordered from highest resolution first.
        """
        x = self.body(x)  # backbone
        y: Dict[str, Tensor] = self.fpn(x)  # FPN
        return y


def _resnet_fpn_extractor(
    backbone: resnet.ResNet,
    spatial_dims: int,
    trainable_layers: int = 5,
    returned_layers: Optional[List[int]] = None,
    extra_blocks: Optional[ExtraFPNBlock] = None,
) -> BackboneWithFPN:
    """
    Same code as https://github.com/pytorch/vision/blob/release/0.12/torchvision/models/detection/backbone_utils.py
    Except that ``in_channels_stage2 = backbone.in_planes // 8`` instead of ``in_channels_stage2 = backbone.inplanes // 8``,
    and it requires spatial_dims: 2D or 3D images.
    """

    # select layers that wont be frozen
    if trainable_layers < 0 or trainable_layers > 5:
        raise ValueError(f"Trainable layers should be in the range [0,5], got {trainable_layers}")
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append("bn1")
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool(spatial_dims)

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    if min(returned_layers) <= 0 or max(returned_layers) >= 5:
        raise ValueError(f"Each returned layer should be in the range [1,4]. Got {returned_layers}")
    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.in_planes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN(
        backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks, spatial_dims=spatial_dims
    )
