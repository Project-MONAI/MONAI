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

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from monai.utils import pytorch_after


def get_mean_kernel_2d(ksize: int = 3) -> torch.Tensor:
    mean_kernel = torch.ones([ksize, ksize]) / (ksize**2)
    return mean_kernel


def get_mean_kernel_3d(ksize: int = 3) -> torch.Tensor:
    mean_kernel = torch.ones([ksize, ksize, ksize]) / (ksize**3)
    return mean_kernel


def get_gaussian_kernel_2d(ksize: int = 3, sigma: float = 1.0) -> torch.Tensor:
    x_grid = torch.arange(ksize).repeat(ksize).view(ksize, ksize)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (ksize - 1) / 2.0
    variance = sigma**2.0
    gaussian_kernel: torch.Tensor = (1.0 / (2.0 * math.pi * variance + 1e-16)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2 * variance + 1e-16)
    )
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    return gaussian_kernel


def get_gaussian_kernel_3d(ksize: int = 3, sigma: float = 1.0) -> torch.Tensor:
    x_coord = torch.arange(ksize)
    x_grid_2d = x_coord.repeat(ksize).view(ksize, ksize)
    x_grid = x_coord.repeat(ksize * ksize).view(ksize, ksize, ksize)
    y_grid_2d = x_grid_2d.t()
    y_grid = y_grid_2d.repeat(ksize, 1).view(ksize, ksize, ksize)
    z_grid = y_grid_2d.repeat(1, ksize).view(ksize, ksize, ksize)
    xyz_grid = torch.stack([x_grid, y_grid, z_grid], dim=-1).float()
    mean = (ksize - 1) / 2.0
    variance = sigma**2.0
    gaussian_kernel: torch.Tensor = (1.0 / (2.0 * math.pi * variance + 1e-16)) * torch.exp(
        -torch.sum((xyz_grid - mean) ** 2.0, dim=-1) / (2 * variance + 1e-16)
    )
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    return gaussian_kernel


class GaussianFilter(torch.nn.Module):
    def __init__(self, dim: int = 3, ksize: int = 3, sigma: float = 1.0, channels: int = 0) -> None:
        super(GaussianFilter, self).__init__()

        self.svls_kernel: torch.Tensor
        self.svls_layer: Any

        if dim == 2:
            gkernel = get_gaussian_kernel_2d(ksize=ksize, sigma=sigma)
            neighbors_sum = (1 - gkernel[1, 1]) + 1e-16
            gkernel[int(ksize / 2), int(ksize / 2)] = neighbors_sum
            self.svls_kernel = gkernel / neighbors_sum

            svls_kernel_2d = self.svls_kernel.view(1, 1, ksize, ksize)
            svls_kernel_2d = svls_kernel_2d.repeat(channels, 1, 1, 1)
            padding = int(ksize / 2)

            self.svls_layer = torch.nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=ksize,
                groups=channels,
                bias=False,
                padding=padding,
                padding_mode="replicate",
            )
            self.svls_layer.weight.data = svls_kernel_2d
            self.svls_layer.weight.requires_grad = False

        if dim == 3:
            gkernel = get_gaussian_kernel_3d(ksize=ksize, sigma=sigma)
            neighbors_sum = 1 - gkernel[1, 1, 1]
            gkernel[1, 1, 1] = neighbors_sum
            self.svls_kernel = gkernel / neighbors_sum

            svls_kernel_3d = self.svls_kernel.view(1, 1, ksize, ksize, ksize)
            svls_kernel_3d = svls_kernel_3d.repeat(channels, 1, 1, 1, 1)
            padding = int(ksize / 2)

            self.svls_layer = torch.nn.Conv3d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=ksize,
                groups=channels,
                bias=False,
                padding=padding,
                padding_mode="replicate",
            )
            self.svls_layer.weight.data = svls_kernel_3d
            self.svls_layer.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        svls_normalized: torch.Tensor = self.svls_layer(x) / self.svls_kernel.sum()
        return svls_normalized


class MeanFilter(torch.nn.Module):
    def __init__(self, dim: int = 3, ksize: int = 3, channels: int = 0) -> None:
        super(MeanFilter, self).__init__()

        self.svls_kernel: torch.Tensor
        self.svls_layer: Any

        if dim == 2:
            self.svls_kernel = get_mean_kernel_2d(ksize=ksize)
            svls_kernel_2d = self.svls_kernel.view(1, 1, ksize, ksize)
            svls_kernel_2d = svls_kernel_2d.repeat(channels, 1, 1, 1)
            padding = int(ksize / 2)

            self.svls_layer = torch.nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=ksize,
                groups=channels,
                bias=False,
                padding=padding,
                padding_mode="replicate",
            )
            self.svls_layer.weight.data = svls_kernel_2d
            self.svls_layer.weight.requires_grad = False

        if dim == 3:
            self.svls_kernel = get_mean_kernel_3d(ksize=ksize)
            svls_kernel_3d = self.svls_kernel.view(1, 1, ksize, ksize, ksize)
            svls_kernel_3d = svls_kernel_3d.repeat(channels, 1, 1, 1, 1)
            padding = int(ksize / 2)

            self.svls_layer = torch.nn.Conv3d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=ksize,
                groups=channels,
                bias=False,
                padding=padding,
                padding_mode="replicate",
            )
            self.svls_layer.weight.data = svls_kernel_3d
            self.svls_layer.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        svls_normalized: torch.Tensor = self.svls_layer(x) / self.svls_kernel.sum()
        return svls_normalized


class NACLLoss(_Loss):
    """
    Neighbor-Aware Calibration Loss (NACL) is primarily developed for developing calibrated models in image segmentation.
    NACL computes standard cross-entropy loss with a linear penalty that enforces the logit distributions
    to match a soft class proportion of surrounding pixel.

    Murugesan, Balamurali, et al.
    "Trust your neighbours: Penalty-based constraints for model calibration."
    International Conference on Medical Image Computing and Computer-Assisted Intervention, MICCAI 2023.
    https://arxiv.org/abs/2303.06268
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
            raise ValueError("Supoorts 2d and 3d")

        if distance_type not in ["l1", "l2"]:
            raise ValueError("Distance type must be either L1 or L2")

        self.nc = classes
        self.dim = dim
        self.cross_entropy = nn.CrossEntropyLoss()
        self.distance_type = distance_type
        self.alpha = alpha
        self.ks = kernel_size
        self.svls_layer: Any

        if kernel_ops == "mean":
            self.svls_layer = MeanFilter(dim=dim, ksize=kernel_size, channels=classes)
        if kernel_ops == "gaussian":
            self.svls_layer = GaussianFilter(dim=dim, ksize=kernel_size, sigma=sigma, channels=classes)

        self.old_pt_ver = not pytorch_after(1, 10)

    # def ce(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #     """
    #     Compute CrossEntropy loss for the input logits and target.
    #     Will remove the channel dim according to PyTorch CrossEntropyLoss:
    #     https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?#torch.nn.CrossEntropyLoss.

    #     """
    #     n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
    #     if n_pred_ch != n_target_ch and n_target_ch == 1:
    #         target = torch.squeeze(target, dim=1)
    #         target = target.long()
    #     elif self.old_pt_ver:
    #         warnings.warn(
    #             f"Multichannel targets are not supported in this older Pytorch version {torch.__version__}. "
    #             "Using argmax (as a workaround) to convert target to a single channel."
    #         )
    #         target = torch.argmax(target, dim=1)
    #     elif not torch.is_floating_point(target):
    #         target = target.to(dtype=input.dtype)

    #     return self.cross_entropy(input, target)  # type: ignore[no-any-return]

    def get_constr_target(self, mask: torch.Tensor) -> torch.Tensor:

        rmask: torch.Tensor

        if self.dim == 2:
            oh_labels = F.one_hot(mask.to(torch.int64), num_classes=self.nc).contiguous().permute(0, 3, 1, 2).float()
            rmask = self.svls_layer(oh_labels)

        if self.dim == 3:
            oh_labels = F.one_hot(mask.to(torch.int64), num_classes=self.nc).contiguous().permute(0, 4, 1, 2, 3).float()
            rmask = self.svls_layer(oh_labels)

        return rmask

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce = self.cross_entropy(inputs, targets)

        utargets = self.get_constr_target(targets)

        if self.distance_type == "l1":
            loss_conf = utargets.sub(inputs).abs_().mean()
        elif self.distance_type == "l2":
            loss_conf = utargets.sub(inputs).pow_(2).abs_().mean()

        loss: torch.Tensor = loss_ce + self.alpha * loss_conf

        return loss
