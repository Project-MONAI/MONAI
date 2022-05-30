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

import math
from typing import Callable, Dict, List, Sequence, Union

import torch
from torch import Tensor, nn

from monai.networks.blocks.backbone_fpn_utils import _resnet_fpn_extractor
from monai.networks.layers.factories import Conv
from monai.networks.nets import resnet
from monai.utils import ensure_tuple_rep, look_up_option, optional_import

_validate_trainable_layers, _ = optional_import(
    "torchvision.models.detection.backbone_utils", name="_validate_trainable_layers"
)


class RetinaNetClassificationHead(nn.Module):
    """
    A classification head for use in RetinaNet.

    This head takes a list of feature maps as inputs, and outputs a list of classification maps.
    Each output map has same spatial size with the corresponding input feature map,
    and the number of output channel is num_anchors * num_classes.

    Args:
        in_channels: number of channels of the input feature
        num_anchors: number of anchors to be predicted
        num_classes: number of classes to be predicted
        spatial_dims: spatial dimension of the network, should be 2 or 3.
        prior_probability: prior probability to initialize classification convolutional layers.
    """

    def __init__(
        self, in_channels: int, num_anchors: int, num_classes: int, spatial_dims: int, prior_probability: float = 0.01
    ):
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        conv = []
        for _ in range(4):
            conv.append(conv_type(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.GroupNorm(num_groups=8, num_channels=in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        for layer in self.conv.children():
            if isinstance(layer, conv_type):  # type: ignore
                torch.nn.init.normal_(layer.weight, std=0.01)  # type: ignore
                torch.nn.init.constant_(layer.bias, 0)  # type: ignore

        self.cls_logits = conv_type(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability))

        self.num_classes = num_classes
        self.num_anchors = num_anchors

    def forward(self, x: Union[List[Tensor], Tensor]) -> List[Tensor]:
        """
        It takes a list of feature maps as inputs, and outputs a list of classification maps.
        Each output classification map has same spatial size with the corresponding input feature map,
        and the number of output channel is num_anchors * num_classes.

        Args:
            x: list of feature map, x[i] is a (B, in_channels, H_i, W_i) or (B, in_channels, H_i, W_i, D_i) Tensor.

        Return:
            cls_logits_maps, list of classification map. cls_logits_maps[i] is a
            (B, num_anchors * num_classes, H_i, W_i) or (B, num_anchors * num_classes, H_i, W_i, D_i) Tensor.

        """
        cls_logits_maps = []

        if isinstance(x, Tensor):
            feature_maps = [x]
        else:
            feature_maps = x

        for features in feature_maps:
            cls_logits = self.conv(features)
            cls_logits = self.cls_logits(cls_logits)

            cls_logits_maps.append(cls_logits)

            if torch.isnan(cls_logits).any() or torch.isinf(cls_logits).any():
                raise ValueError("cls_logits is NaN or Inf.")

        return cls_logits_maps


class RetinaNetRegressionHead(nn.Module):
    """
    A regression head for use in RetinaNet.

    This head takes a list of feature maps as inputs, and outputs a list of box regression maps.
    Each output box regression map has same spatial size with the corresponding input feature map,
    and the number of output channel is num_anchors * 2 * spatial_dims.

    Args:
        in_channels: number of channels of the input feature
        num_anchors: number of anchors to be predicted
        spatial_dims: spatial dimension of the network, should be 2 or 3.
    """

    def __init__(self, in_channels: int, num_anchors: int, spatial_dims: int):
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]

        conv = []
        for _ in range(4):
            conv.append(conv_type(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.GroupNorm(num_groups=8, num_channels=in_channels))
            conv.append(nn.ReLU())

        self.conv = nn.Sequential(*conv)

        self.bbox_reg = conv_type(in_channels, num_anchors * 2 * spatial_dims, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.bbox_reg.weight, std=0.01)
        torch.nn.init.zeros_(self.bbox_reg.bias)

        for layer in self.conv.children():
            if isinstance(layer, conv_type):  # type: ignore
                torch.nn.init.normal_(layer.weight, std=0.01)  # type: ignore
                torch.nn.init.zeros_(layer.bias)  # type: ignore

    def forward(self, x: Union[List[Tensor], Tensor]) -> List[Tensor]:
        """
        It takes a list of feature maps as inputs, and outputs a list of box regression maps.
        Each output box regression map has same spatial size with the corresponding input feature map,
        and the number of output channel is num_anchors * 2 * spatial_dims.

        Args:
            x: list of feature map, x[i] is a (B, in_channels, H_i, W_i) or (B, in_channels, H_i, W_i, D_i) Tensor.

        Return:
            box_regression_maps, list of box regression map. cls_logits_maps[i] is a
            (B, num_anchors * 2 * spatial_dims, H_i, W_i) or (B, num_anchors * 2 * spatial_dims, H_i, W_i, D_i) Tensor.

        """
        box_regression_maps = []

        if isinstance(x, Tensor):
            feature_maps = [x]
        else:
            feature_maps = x

        for features in feature_maps:
            box_regression = self.conv(features)
            box_regression = self.bbox_reg(box_regression)

            box_regression_maps.append(box_regression)

            if torch.isnan(box_regression).any() or torch.isinf(box_regression).any():
                raise ValueError("box_regression is NaN or Inf.")

        return box_regression_maps


class RetinaNet(nn.Module):
    """
    The network used in RetinaNet.

    It takes an image tensor as inputs, and outputs a dictionary ``head_outputs``.
    ``head_outputs[self.cls_key]`` is the predicted classification maps, a list of Tensor.
    ``head_outputs[self.box_reg_key]`` is the predicted box regression maps, a list of Tensor.

    Args:
        spatial_dims: number of spatial dimensions of the images. We support both 2D and 3D images.
        num_classes: number of output classes of the model (excluding the background).
        num_anchors: number of anchors at each location.
        feature_extractor: a network that outputs feature maps from the input images,
            each feature map corresponds to a different resolution.
            Its output can have format of Tensor, Dict[Any, Tensor], or Sequence[Tensor].
            It can be the output of ``resnet_fpn_feature_extractor(*args, **kwargs)``.
        size_divisible: the spatial size of the network input should be divisible by size_divisible,
            decided by the feature_extractor.

    Example:

        .. code-block:: python

            from monai.networks.nets import resnet
            spatial_dims = 3  # 3D network
            conv1_t_stride = (2,2,1)  # stride of first convolutional layer in backbone
            backbone = resnet.ResNet(
                spatial_dims = spatial_dims,
                block = resnet.ResNetBottleneck,
                layers = [3, 4, 6, 3],
                block_inplanes = resnet.get_inplanes(),
                n_input_channels= 1,
                conv1_t_stride = conv1_t_stride,
                conv1_t_size = (7,7,7),
            )
            # This feature_extractor outputs 4-level feature maps.
            # number of output feature maps is len(returned_layers)+1
            returned_layers = [1,2,3]  # returned layer from feature pyramid network
            feature_extractor = resnet_fpn_feature_extractor(
                backbone = backbone,
                spatial_dims = spatial_dims,
                pretrained_backbone = False,
                trainable_backbone_layers = None,
                returned_layers = returned_layers,
            )
            # This feature_extractor requires input imgage spatial size
            # to be divisible by (32, 32, 16).
            size_divisible = tuple(2*s*2**max(returned_layers) for s in conv1_t_stride)
            model = RetinaNet(
                spatial_dims = spatial_dims,
                num_classes = 5,
                num_anchors = 6,
                feature_extractor=feature_extractor,
                size_divisible = size_divisible,
            ).to(device)
            result = model(torch.rand(2, 1, 128,128,128))
            cls_logits_maps = result["cls_logits"]  # a list of len(returned_layers)+1 Tensor
            box_regression_maps = result["box_regression"]  # a list of len(returned_layers)+1 Tensor
    """

    def __init__(
        self,
        spatial_dims: int,
        num_classes: int,
        num_anchors: int,
        feature_extractor,
        size_divisible: Union[Sequence[int], int] = 1,
    ):
        super().__init__()

        self.spatial_dims = look_up_option(spatial_dims, supported=[1, 2, 3])
        self.num_classes = num_classes
        self.size_divisible = ensure_tuple_rep(size_divisible, self.spatial_dims)

        if not hasattr(feature_extractor, "out_channels"):
            raise ValueError(
                "feature_extractor should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )
        self.feature_extractor = feature_extractor

        self.feature_map_channels: int = self.feature_extractor.out_channels
        self.num_anchors = num_anchors
        self.classification_head = RetinaNetClassificationHead(
            self.feature_map_channels, self.num_anchors, self.num_classes, spatial_dims=self.spatial_dims
        )
        self.regression_head = RetinaNetRegressionHead(
            self.feature_map_channels, self.num_anchors, spatial_dims=self.spatial_dims
        )

        self.cls_key: str = "classification"
        self.box_reg_key: str = "box_regression"

    def forward(self, images: Tensor) -> Dict[str, List[Tensor]]:
        """
        It takes an image tensor as inputs, and outputs a dictionary ``head_outputs``.
        ``head_outputs[self.cls_key]`` is the predicted classification maps, a list of Tensor.
        ``head_outputs[self.box_reg_key]`` is the predicted box regression maps, a list of Tensor.

        Args:
            images: input images, sized (B, img_channels, H, W) or (B, img_channels, H, W, D).

        Return:
            a dictionary ``head_outputs`` with keys including self.cls_key and self.box_reg_key.
            ``head_outputs[self.cls_key]`` is the predicted classification maps, a list of Tensor.
            ``head_outputs[self.box_reg_key]`` is the predicted box regression maps, a list of Tensor.

        """
        # compute features maps list from the input images.
        features = self.feature_extractor(images)
        if isinstance(features, Tensor):
            feature_maps = [features]
        elif torch.jit.isinstance(features, Dict[str, Tensor]):
            feature_maps = list(features.values())
        else:
            feature_maps = list(features)

        if not isinstance(feature_maps[0], Tensor):
            raise ValueError("feature_extractor output format must be Tensor, Dict[str, Tensor], or Sequence[Tensor].")

        # compute classification and box regression maps from the feature maps
        # expandable for mask prediction in the future

        head_outputs: Dict[str, List[Tensor]] = {self.cls_key: self.classification_head(feature_maps)}
        head_outputs[self.box_reg_key] = self.regression_head(feature_maps)

        return head_outputs


def resnet_fpn_feature_extractor(
    backbone: resnet.ResNet,
    spatial_dims: int,
    pretrained_backbone: bool = False,
    returned_layers: Sequence[int] = (1, 2, 3),
    trainable_backbone_layers: Union[int, None] = None,
):
    """
    Constructs a feature extractor network with a ResNet-FPN backbone, used as feature_extractor in RetinaNet.

    Reference: `"Focal Loss for Dense Object Detection" <https://arxiv.org/abs/1708.02002>`_.

    The returned feature_extractor network takes an image tensor as inputs,
    and outputs a dictionary that maps string to the extracted feature maps (Tensor).

    The input to the returned feature_extractor is expected to be a list of tensors,
    each of shape ``[C, H, W]`` or ``[C, H, W, D]``,
    one for each image. Different images can have different sizes.


    Args:
        backbone: a ResNet model, used as backbone.
        spatial_dims: number of spatial dimensions of the images. We support both 2D and 3D images.
        pretrained_backbone: whether the backbone has been pre-trained.
        returned_layers: returned layers to extract feature maps. Each returned layer should be in the range [1,4].
            len(returned_layers)+1 will be the number of extracted feature maps.
            There is an extra maxpooling layer LastLevelMaxPool() appended.
        trainable_backbone_layers: number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
            When pretrained_backbone is False, this value is set to be 5.
            When pretrained_backbone is True, if ``None`` is passed (the default) this value is set to 3.

    Example:

        .. code-block:: python

            from monai.networks.nets import resnet
            spatial_dims = 3 # 3D network
            backbone = resnet.ResNet(
                spatial_dims = spatial_dims,
                block = resnet.ResNetBottleneck,
                layers = [3, 4, 6, 3],
                block_inplanes = resnet.get_inplanes(),
                n_input_channels= 1,
                conv1_t_stride = (2,2,1),
                conv1_t_size = (7,7,7),
            )
            # This feature_extractor outputs 4-level feature maps.
            # number of output feature maps is len(returned_layers)+1
            feature_extractor = resnet_fpn_feature_extractor(
                backbone = backbone,
                spatial_dims = spatial_dims,
                pretrained_backbone = False,
                trainable_backbone_layers = None,
                returned_layers = [1,2,3],
            )
            model = RetinaNet(
                spatial_dims = spatial_dims,
                num_classes = 5,
                num_anchors = 6,
                feature_extractor=feature_extractor,
                size_divisible = 32,
            ).to(device)
    """
    # If pretrained_backbone is False, valid_trainable_backbone_layers = 5.
    # If pretrained_backbone is True, valid_trainable_backbone_layers = trainable_backbone_layers or 3 if None.
    valid_trainable_backbone_layers: int = _validate_trainable_layers(
        pretrained_backbone, trainable_backbone_layers, max_value=5, default_value=3
    )

    feature_extractor = _resnet_fpn_extractor(
        backbone,
        spatial_dims,
        valid_trainable_backbone_layers,
        returned_layers=list(returned_layers),
        extra_blocks=None,
    )
    return feature_extractor
