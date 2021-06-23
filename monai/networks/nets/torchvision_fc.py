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

from typing import Tuple, Union

from monai.networks.nets import FinetuneFC
from monai.utils import optional_import

models, _ = optional_import("torchvision.models")


__all__ = ["TorchVisionFCModel"]


class TorchVisionFCModel(FinetuneFC):
    """
    Customize the fully connected layer of TorchVision model or replace it by convolutional layer.

    Args:
        model_name: name of any torchvision model with fully connected layer at the end.
            ``resnet18`` (default), ``resnet34m``, ``resnet50``, ``resnet101``, ``resnet152``,
            ``resnext50_32x4d``, ``resnext101_32x8d``, ``wide_resnet50_2``, ``wide_resnet101_2``.
            model details: https://pytorch.org/vision/stable/models.html.
        n_classes: number of classes for the last classification layer. Default to 1.
        use_conv: whether use convolutional layer to replace the FC layer, default to False.
        pool_size: if using convolutional layer to replace the FC layer, it defines the kernel size for `AvgPool2d`
            to replace `AdaptiveAvgPool2d`. Default to (7, 7).
        pool_stride: if using convolutional layer to replace the FC layer, it defines the stride for `AvgPool2d`
            to replace `AdaptiveAvgPool2d`. Default to 1.
        bias: the bias value when replacing FC layer. if False, the layer will not learn an additive bias,
            default to True.
        pretrained: whether to use the imagenet pretrained weights. Default to False.
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        n_classes: int = 1,
        use_conv: bool = False,
        pool_size: Union[int, Tuple[int, int]] = (7, 7),
        pool_stride: Union[int, Tuple[int, int]] = 1,
        pretrained: bool = False,
        bias: bool = True,
    ):
        model = getattr(models, model_name)(pretrained=pretrained)
        super().__init__(
            model=model,
            n_classes=n_classes,
            use_conv=use_conv,
            pool_size=pool_size,
            pool_stride=pool_stride,
            bias=bias,
        )
