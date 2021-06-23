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

from typing import Optional, Tuple, Union

import torch

from monai.networks.layers import Conv, get_pool_layer


class FinetuneFC(torch.nn.Module):
    """
    Wrapper to customize the fully connected layer of a classification model or replace it by convolutional layer.
    This module expects the output of `model layers[0: -2]` is a feature map with shape [B, C, spatial dims],
    then replace the model's last two layers with a `pooling + conv`, or replace the last layer with a `linear`.

    Args:
        model: PyTorch model with fully connected layer at the end, support both 2D and 3D models.
            typically, it can be a pretrained model in Torchvision, like:
            ``resnet18``, ``resnet34m``, ``resnet50``, ``resnet101``, ``resnet152``, etc.
            more details: https://pytorch.org/vision/stable/models.html.
        n_classes: number of classes for the last classification layer. Default to 1.
        dim: number of spatial dimensions, default to 2.
        use_conv: whether use convolutional layer to replace the FC layer, default to False.
        pool_size: if using convolutional layer to replace the FC layer, it defines the kernel size for `AvgPool`,
            default to 7.
        pool_stride: if using convolutional layer to replace the FC layer, it defines the stride for `AvgPool`,
            default to 1.
        bias: the bias value when replacing FC layer. if False, the layer will not learn an additive bias,
            default to True.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        n_classes: int = 1,
        dim: int = 2,
        use_conv: bool = False,
        pool_size: Optional[Union[int, Tuple[int, int]]] = 7,
        pool_stride: Optional[Union[int, Tuple[int, int]]] = 1,
        bias: bool = True,
    ):
        super().__init__()
        layers = list(model.children())

        orig_fc = layers[-1]
        self.fc: Union[torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv3d]
        self.pool = None
        if use_conv:
            # remove the last Linear layer (fully connected) and the adaptive avg pooling
            self.features = torch.nn.Sequential(*layers[:-2])
            if pool_size is not None and pool_stride is not None:
                # add 7x7 avg pooling
                self.pool = get_pool_layer(
                    name=("avg", {"kernel_size": pool_size, "stride": pool_stride}),
                    spatial_dims=dim,
                )
            # add 1x1 conv (it behaves like a FC layer)
            self.fc = Conv[Conv.CONV, dim](
                in_channels=orig_fc.in_features,  # type: ignore
                out_channels=n_classes,
                kernel_size=(1, 1) if dim == 2 else (1, 1, 1),
            )
        else:
            # remove the last Linear layer (fully connected)
            self.features = torch.nn.Sequential(*layers[:-1])
            # replace the out_features of FC layer
            self.fc = torch.nn.Linear(
                in_features=orig_fc.in_features,  # type: ignore
                out_features=n_classes,
                bias=bias,
            )
        self.use_conv = use_conv

    def forward(self, x):
        x = self.features(x)
        if self.pool is not None:
            x = self.pool(x)
        else:
            x = torch.flatten(x, 1)

        x = self.fc(x)

        return x
