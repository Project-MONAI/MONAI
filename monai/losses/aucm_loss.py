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

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

from monai.utils import LossReduction


class AUCMLoss(_Loss):
    """
    AUC-Margin loss with squared-hinge surrogate loss for optimizing AUROC.

    The loss optimizes the Area Under the ROC Curve (AUROC) by using margin-based constraints
    on positive and negative predictions. It supports two versions: 'v1' includes class prior
    information, while 'v2' removes this dependency for better generalization.

    Reference:
        Yuan, Zhuoning, Yan, Yan, Sonka, Milan, and Yang, Tianbao.
        "Large-scale robust deep auc maximization: A new surrogate loss and empirical studies on medical image classification."
        Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
        https://arxiv.org/abs/2012.03173

        Implementation based on: https://github.com/Optimization-AI/LibAUC/blob/1.4.0/libauc/losses/auc.py

    Example:
        >>> import torch
        >>> from monai.losses import AUCMLoss
        >>> loss_fn = AUCMLoss()
        >>> input = torch.randn(32, 1, requires_grad=True)
        >>> target = torch.randint(0, 2, (32, 1)).float()
        >>> loss = loss_fn(input, target)
    """

    def __init__(
        self,
        margin: float = 1.0,
        imratio: float | None = None,
        version: str = "v1",
        reduction: LossReduction | str = LossReduction.MEAN,
    ) -> None:
        """
        Args:
            margin: margin for squared-hinge surrogate loss (default: ``1.0``).
            imratio: the ratio of the number of positive samples to the number of total samples in the training dataset.
                If this value is not given, it will be automatically calculated with mini-batch samples.
                This value is ignored when ``version`` is set to ``'v2'``.
            version: whether to include prior class information in the objective function (default: ``'v1'``).
                'v1' includes class prior, 'v2' removes this dependency.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.
                Note: This loss is computed at the batch level and always returns a scalar.
                The reduction parameter is accepted for API consistency but has no effect.

        Raises:
            ValueError: When ``version`` is not one of ["v1", "v2"].
            ValueError: When ``imratio`` is not in [0, 1].

        Example:
            >>> import torch
            >>> from monai.losses import AUCMLoss
            >>> loss_fn = AUCMLoss(version='v2')
            >>> input = torch.randn(32, 1, requires_grad=True)
            >>> target = torch.randint(0, 2, (32, 1)).float()
            >>> loss = loss_fn(input, target)
        """
        super().__init__(reduction=LossReduction(reduction).value)
        if version not in ["v1", "v2"]:
            raise ValueError(f"version should be 'v1' or 'v2', got {version}")
        if imratio is not None and not (0.0 <= imratio <= 1.0):
            raise ValueError(f"imratio must be in [0, 1], got {imratio}")
        self.margin = margin
        self.imratio = imratio
        self.version = version
        self.a = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be B1HW[D], where the channel dimension is 1 for binary classification.
            target: the shape should be B1HW[D], with values 0 or 1.

        Returns:
            torch.Tensor: scalar AUCM loss.

        Raises:
            ValueError: When input or target have incorrect shapes.
            ValueError: When input or target have fewer than 2 dimensions.
            ValueError: When target contains non-binary values.
        """
        if input.ndim < 2 or target.ndim < 2:
            raise ValueError("Input and target must have at least 2 dimensions (B, C, ...)")
        if input.shape[1] != 1:
            raise ValueError(f"Input should have 1 channel for binary classification, got {input.shape[1]}")
        if target.shape[1] != 1:
            raise ValueError(f"Target should have 1 channel, got {target.shape[1]}")
        if input.shape != target.shape:
            raise ValueError(f"Input and target shapes do not match: {input.shape} vs {target.shape}")

        input = input.flatten()
        target = target.flatten()

        if not torch.all((target == 0) | (target == 1)):
            raise ValueError("Target must contain only binary values (0 or 1)")

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()

        if self.version == "v1":
            p = float(self.imratio) if self.imratio is not None else float(pos_mask.mean().item())
            loss = (
                (1 - p) * self._safe_mean((input - self.a) ** 2, pos_mask)
                + p * self._safe_mean((input - self.b) ** 2, neg_mask)
                + 2
                * self.alpha
                * (
                    p * (1 - p) * self.margin
                    + self._safe_mean(p * input * neg_mask - (1 - p) * input * pos_mask, pos_mask + neg_mask)
                )
                - p * (1 - p) * self.alpha**2
            )
        else:
            loss = (
                self._safe_mean((input - self.a) ** 2, pos_mask)
                + self._safe_mean((input - self.b) ** 2, neg_mask)
                + 2 * self.alpha * (self.margin + self._safe_mean(input, neg_mask) - self._safe_mean(input, pos_mask))
                - self.alpha**2
            )

        return loss

    def _safe_mean(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute mean safely over masked elements."""
        denom = mask.sum()
        if denom == 0:
            return torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype)
        return (tensor * mask).sum() / denom
