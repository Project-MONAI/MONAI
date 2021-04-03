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

import torch

from monai.utils import optional_import

models, _ = optional_import("torchvision.models")


class TorchVisionFullyConvModel(torch.nn.Module):
    """
    Customize TorchVision models to replace fully connected layer by convolutional layer.

    Args:
        model_name: name of any torchvision with adaptive avg pooling and fully connected layer at the end.
            ``resnet18`` (default), ``resnet34m``, ``resnet50``, ``resnet101``, ``resnet152``,
            ``resnext50_32x4d``, ``resnext101_32x8d``, ``wide_resnet50_2``, ``wide_resnet101_2``.
        n_classes: number of classes for the last classification layer. Default to 1.
        pool_size: the kernel size for `AvgPool2d` to replace `AdaptiveAvgPool2d`. Default to (7, 7).
        pool_stride: the stride for `AvgPool2d` to replace `AdaptiveAvgPool2d`. Default to 1.
        pretrained: whether to use the imagenet pretrained weights. Default to False.
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        n_classes: int = 1,
        pool_size: Union[int, Tuple[int, int]] = (7, 7),
        pool_stride: Union[int, Tuple[int, int]] = 1,
        pretrained: bool = False,
    ):
        super().__init__()
        model = getattr(models, model_name)(pretrained=pretrained)
        layers = list(model.children())

        # check if the model is compatible
        if not str(layers[-1]).startswith("Linear"):
            raise ValueError(f"Model ['{model_name}'] does not have a Linear layer at the end.")
        if not str(layers[-2]).startswith("AdaptiveAvgPool2d"):
            raise ValueError(f"Model ['{model_name}'] does not have a AdaptiveAvgPool2d layer next to the end.")

        # remove the last Linear layer (fully connected) and the adaptive avg pooling
        self.features = torch.nn.Sequential(*layers[:-2])

        # add 7x7 avg pooling (in place of adaptive avg pooling)
        self.pool = torch.nn.AvgPool2d(kernel_size=pool_size, stride=pool_stride)

        # add 1x1 conv (it behaves like a FC layer)
        self.fc = torch.nn.Conv2d(model.fc.in_features, n_classes, kernel_size=(1, 1))

    def forward(self, x):
        x = self.features(x)

        # apply 2D avg pooling
        x = self.pool(x)

        # apply last 1x1 conv layer that act like a linear layer
        x = self.fc(x)

        return x
