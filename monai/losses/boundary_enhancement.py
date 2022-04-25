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

import monai
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryEnhancementLoss(nn.Module):
    def __init__(
        self,
        output_classes,
        device,
        num_dims=3,
    ):
        super().__init__()

        self.num_dims = num_dims
        self.output_classes = output_classes

        if self.num_dims == 2:
            self.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
                padding_mode="zeros",
            ).to(device)

            self.conv2 = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
                padding_mode="zeros",
            ).to(device)

            weight = np.zeros(shape=(1, 1, 3, 3), dtype=np.float32)
            weight[..., :, :] = np.array(
                [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32
            )

            with torch.no_grad():
                self.conv1.weight.copy_(
                    torch.from_numpy(
                        1.0 / 9.0 * np.ones(shape=weight.shape, dtype=np.float32)
                    )
                )
                self.conv2.weight.copy_(torch.from_numpy(weight))
        elif self.num_dims == 3:
            self.conv1 = nn.Conv3d(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
                padding_mode="zeros",
            ).to(device)

            self.conv2 = nn.Conv3d(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
                padding_mode="zeros",
            ).to(device)

            weight = np.zeros(shape=(1, 1, 3, 3, 3), dtype=np.float32)
            weight[..., 0, :, :] = np.array(
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32
            )
            weight[..., 1, :, :] = np.array(
                [[0, 1, 0], [1, -6, 1], [0, 1, 0]], dtype=np.float32
            )
            weight[..., 2, :, :] = np.array(
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32
            )

            with torch.no_grad():
                self.conv1.weight.copy_(
                    torch.from_numpy(
                        1.0 / 27.0 * np.ones(shape=weight.shape, dtype=np.float32)
                    )
                )
                self.conv2.weight.copy_(torch.from_numpy(weight))
        else:
            print("[error] number of dimensions is incorrect!")

        self.conv1.weight.requires_grad = False
        self.conv2.weight.requires_grad = False

    def compute_boundary(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

    def forward(
        self, output, target, weight=1.0, to_onehot=True, include_background=False
    ):
        output = F.softmax(output, dim=1)
        boundary_loss = 0

        if to_onehot:
            if self.num_dims == 2:
                target = torch.permute(target, (1, 2, 3, 0))
                target = monai.transforms.AsDiscrete(to_onehot=self.output_classes)(
                    target
                )
                target = torch.permute(target, (3, 0, 1, 2))
            elif self.num_dims == 3:
                target = torch.permute(target, (1, 2, 3, 4, 0))
                target = monai.transforms.AsDiscrete(to_onehot=self.output_classes)(
                    target
                )
                target = torch.permute(target, (4, 0, 1, 2, 3))

        if include_background:
            starting_class_index = 0
        else:
            starting_class_index = 1

        for _i in range(starting_class_index, self.output_classes):
            output_boundary = self.compute_boundary(output[:, _i : _i + 1, ...])
            target_boundary = self.compute_boundary(target[:, _i : _i + 1, ...])
            loss_value = torch.square(output_boundary - target_boundary)
            boundary_loss += loss_value

        boundary_loss = boundary_loss.mean()
        return weight * boundary_loss
