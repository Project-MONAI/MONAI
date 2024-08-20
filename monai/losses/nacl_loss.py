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

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from monai.networks.layers import GaussianFilter, MeanFilter


class NACLLoss(_Loss):
    """
    Neighbor-Aware Calibration Loss (NACL) is primarily developed for developing calibrated models in image segmentation.
    NACL computes standard cross-entropy loss with a linear penalty that enforces the logit distributions
    to match a soft class proportion of surrounding pixel.

    Murugesan, Balamurali, et al.
    "Trust your neighbours: Penalty-based constraints for model calibration."
    International Conference on Medical Image Computing and Computer-Assisted Intervention, MICCAI 2023.
    https://arxiv.org/abs/2303.06268

    Murugesan, Balamurali, et al.
    "Neighbor-Aware Calibration of Segmentation Networks with Penalty-Based Constraints."
    https://arxiv.org/abs/2401.14487
    """

    def __init__(
        self,
        classes: int,
        dim: int,
        kernel_size: int = 3,
        kernel_ops: str = "mean",
        distance_type: str = "l1",
        alpha: float = 0.1,
        sigma: float = 1.0,
    ) -> None:
        """
        Args:
            classes: number of classes
            dim: dimension of data (supports 2d and 3d)
            kernel_size: size of the spatial kernel
            distance_type: l1/l2 distance between spatial kernel and predicted logits
            alpha: weightage between cross entropy and logit constraint
            sigma: sigma of gaussian
        """

        super().__init__()

        if kernel_ops not in ["mean", "gaussian"]:
            raise ValueError("Kernel ops must be either mean or gaussian")

        if dim not in [2, 3]:
            raise ValueError(f"Support 2d and 3d, got dim={dim}.")

        if distance_type not in ["l1", "l2"]:
            raise ValueError(f"Distance type must be either L1 or L2, got {distance_type}")

        self.nc = classes
        self.dim = dim
        self.cross_entropy = nn.CrossEntropyLoss()
        self.distance_type = distance_type
        self.alpha = alpha
        self.ks = kernel_size
        self.svls_layer: Any

        if kernel_ops == "mean":
            self.svls_layer = MeanFilter(spatial_dims=dim, size=kernel_size)
            self.svls_layer.filter = self.svls_layer.filter / (kernel_size**dim)
        if kernel_ops == "gaussian":
            self.svls_layer = GaussianFilter(spatial_dims=dim, sigma=sigma)

    def get_constr_target(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Converts the mask to one hot represenation and is smoothened with the selected spatial filter.

        Args:
            mask: the shape should be BH[WD].

        Returns:
            torch.Tensor: the shape would be BNH[WD], N being number of classes.
        """
        rmask: torch.Tensor

        if self.dim == 2:
            oh_labels = F.one_hot(mask.to(torch.int64), num_classes=self.nc).permute(0, 3, 1, 2).contiguous().float()
            rmask = self.svls_layer(oh_labels)

        if self.dim == 3:
            oh_labels = F.one_hot(mask.to(torch.int64), num_classes=self.nc).permute(0, 4, 1, 2, 3).contiguous().float()
            rmask = self.svls_layer(oh_labels)

        return rmask

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes standard cross-entropy loss and constraints it neighbor aware logit penalty.

        Args:
            inputs: the shape should be BNH[WD], where N is the number of classes.
            targets: the shape should be BH[WD].

        Returns:
            torch.Tensor: value of the loss.

        Example:
            >>> import torch
            >>> from monai.losses import NACLLoss
            >>> B, N, H, W = 8, 3, 64, 64
            >>> input = torch.rand(B, N, H, W)
            >>> target = torch.randint(0, N, (B, H, W))
            >>> criterion = NACLLoss(classes = N, dim = 2)
            >>> loss = criterion(input, target)
        """

        loss_ce = self.cross_entropy(inputs, targets)

        utargets = self.get_constr_target(targets)

        if self.distance_type == "l1":
            loss_conf = utargets.sub(inputs).abs_().mean()
        elif self.distance_type == "l2":
            loss_conf = utargets.sub(inputs).pow_(2).abs_().mean()

        loss: torch.Tensor = loss_ce + self.alpha * loss_conf

        return loss
