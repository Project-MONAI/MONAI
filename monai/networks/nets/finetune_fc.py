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


class FinetuneFC(torch.nn.Module):
    """
    Wrapper to customize the fully connected layer of 2D classification model or replace it by convolutional layer.

    Args:
        model: PyTorch model with fully connected layer at the end, typically, it can be a pretrained model in
            Torchvision, like: ``resnet18``, ``resnet34m``, ``resnet50``, ``resnet101``, ``resnet152``, etc.
            model details: https://pytorch.org/vision/stable/models.html.
        n_classes: number of classes for the last classification layer. Default to 1.
        use_conv: whether use convolutional layer to replace the FC layer, default to False.
        pool_size: if using convolutional layer to replace the FC layer, it defines the kernel size for `AvgPool2d`
            to replace `AdaptiveAvgPool2d`. Default to (7, 7).
        pool_stride: if using convolutional layer to replace the FC layer, it defines the stride for `AvgPool2d`
            to replace `AdaptiveAvgPool2d`. Default to 1.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        n_classes: int = 1,
        use_conv: bool = False,
        pool_size: Union[int, Tuple[int, int]] = (7, 7),
        pool_stride: Union[int, Tuple[int, int]] = 1,
    ):
        super().__init__()
        layers = list(model.children())

        # check if the model is compatible
        if not str(layers[-1]).startswith("Linear"):
            raise ValueError("input model does not have a Linear layer at the end.")
        orig_fc = layers[-1]
        self.fc: Union[torch.nn.Linear, torch.nn.Conv2d]
        if use_conv:
            if not str(layers[-2]).startswith("AdaptiveAvgPool2d"):
                raise ValueError("input model does not have a AdaptiveAvgPool2d layer next to the end.")

            # remove the last Linear layer (fully connected) and the adaptive avg pooling
            self.features = torch.nn.Sequential(*layers[:-2])
            # add 7x7 avg pooling (in place of adaptive avg pooling)
            self.pool = torch.nn.AvgPool2d(kernel_size=pool_size, stride=pool_stride)
            # add 1x1 conv (it behaves like a FC layer)
            self.fc = torch.nn.Conv2d(
                in_channels=orig_fc.in_features,  # type: ignore
                out_channels=n_classes,
                kernel_size=(1, 1),
            )
        else:
            # remove the last Linear layer (fully connected)
            self.features = torch.nn.Sequential(*layers[:-1])
            # replace the out_features of FC layer
            self.fc = torch.nn.Linear(
                in_features=orig_fc.in_features,  # type: ignore
                out_features=n_classes,
                bias=True,
            )
        self.use_conv = use_conv

    def forward(self, x):
        x = self.features(x)
        if self.use_conv:
            # apply 2D avg pooling
            x = self.pool(x)
        else:
            x = torch.flatten(x, 1)

        x = self.fc(x)

        return x
