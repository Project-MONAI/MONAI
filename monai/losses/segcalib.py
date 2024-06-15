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
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from monai.utils import pytorch_after


def get_gaussian_kernel_2d(ksize: int = 3, sigma: float = 1.0) -> torch.Tensor:
    x_grid = torch.arange(ksize).repeat(ksize).view(ksize, ksize)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (ksize - 1) / 2.0
    variance = sigma**2.0
    gaussian_kernel = (1.0 / (2.0 * math.pi * variance + 1e-16)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2 * variance + 1e-16)
    )
    return gaussian_kernel / torch.sum(gaussian_kernel)


class GaussianFilter(torch.nn.Module):
    def __init__(self, ksize: int = 3, sigma: float = 1.0, channels: int = 0) -> torch.Tensor:
        super(GaussianFilter, self).__init__()
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

    def forward(self, x):
        return self.svls_layer(x) / self.svls_kernel.sum()


class NACLLoss(_Loss):
    """
    Neighbor-Aware Calibration Loss (NACL) is primarily developed for developing calibrated models in image segmentation.
    NACL computes standard cross-entropy loss with a linear penalty that enforces the logit distributions
    to match a soft class proportion of surrounding pixel. 

    Murugesan, Balamurali, et al.
    "Trust your neighbours: Penalty-based constraints for model calibration."
    International Conference on Medical Image Computing and Computer-Assisted Intervention, 2023.
    https://arxiv.org/abs/2303.06268
    """

    def __init__(
        self,
        classes,
        kernel_size: int = 3,
        kernel_ops: str = "mean",
        distance_type: str = "l1",
        alpha: float = 0.1,
        sigma: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            classes: number of classes
            kernel_size: size of the spatial kernel
            kenel_ops: type of kernel operation (mean/gaussian)
            distance_type: l1/l2 distance between spatial kernel and predicted logits
            alpha: weightage between cross entropy and logit constraint
            sigma: sigma if the kernel type is gaussian
        """

        super().__init__()

        if kernel_ops not in ["mean", "gaussian"]:
            raise ValueError("Kernel ops must be either mean or gaussian")

        if distance_type not in ["l1", "l2"]:
            raise ValueError("Distance type must be either L1 or L2")

        self.kernel_ops = kernel_ops
        self.distance_type = distance_type
        self.alpha = alpha

        self.nc = classes
        self.ks = kernel_size
        self.cross_entropy = nn.CrossEntropyLoss()

        if kernel_ops == "gaussian":
            self.svls_layer = GaussianFilter(ksize=kernel_size, sigma=sigma, channels=classes)

        self.old_pt_ver = not pytorch_after(1, 10)

    def ce(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute CrossEntropy loss for the input logits and target.
        Will remove the channel dim according to PyTorch CrossEntropyLoss:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?#torch.nn.CrossEntropyLoss.

        """
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch != n_target_ch and n_target_ch == 1:
            target = torch.squeeze(target, dim=1)
            target = target.long()
        elif self.old_pt_ver:
            warnings.warn(
                f"Multichannel targets are not supported in this older Pytorch version {torch.__version__}. "
                "Using argmax (as a workaround) to convert target to a single channel."
            )
            target = torch.argmax(target, dim=1)
        elif not torch.is_floating_point(target):
            target = target.to(dtype=input.dtype)

        return self.cross_entropy(input, target)  # type: ignore[no-any-return]

    def get_constr_target(self, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(1)  # unfold works for 4d.

        bs, _, h, w = mask.shape
        unfold = torch.nn.Unfold(kernel_size=(self.ks, self.ks), padding=self.ks // 2)

        rmask = []

        if self.kernel_ops == "mean":
            umask = unfold(mask.float())

            for ii in range(self.nc):
                rmask.append(torch.sum(umask == ii, 1) / self.ks**2)

        if self.kernel_ops == "gaussian":
            oh_labels = (
                F.one_hot(mask[:, 0].to(torch.int64), num_classes=self.nc).contiguous().permute(0, 3, 1, 2).float()
            )
            rmask = self.svls_layer(oh_labels)

            return rmask

        rmask = torch.stack(rmask, dim=1)
        rmask = rmask.reshape(bs, self.nc, h, w)

        return rmask

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce(inputs, targets)

        utargets = self.get_constr_target(targets)

        if self.distance_type == "l1":
            loss_conf = utargets.sub(inputs).abs_().mean()
        elif self.distance_type == "l2":
            loss_conf = utargets.sub(inputs).pow_(2).abs_().mean()

        loss = loss_ce + self.alpha * loss_conf

        return loss
