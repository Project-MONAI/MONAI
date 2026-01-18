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

import warnings

import torch
from torch.nn.modules.loss import _Loss

from monai.networks import one_hot
from monai.utils import LossReduction


class AsymmetricUnifiedFocalLoss(_Loss):
    """
    AsymmetricUnifiedFocalLoss is a variant of Focal Loss that combines Asymmetric Focal Loss
    and Asymmetric Focal Tversky Loss to handle imbalanced medical image segmentation.

    It supports multi-class segmentation by treating channel 0 as background and
    channels 1..N as foreground, applying asymmetric weighting controlled by `delta`.

    Reimplementation of the Asymmetric Unified Focal Loss described in:

    - "Unified Focal Loss: Generalising Dice and Cross Entropy-based Losses to Handle Class Imbalanced Medical Image Segmentation",
    Michael Yeung, Computerized Medical Imaging and Graphics

    Example:
        >>> import torch
        >>> from monai.losses import AsymmetricUnifiedFocalLoss
        >>> # B, C, H, W = 1, 3, 32, 32
        >>> pred_logits = torch.randn(1, 3, 32, 32)
        >>> # Ground truth indices (B, 1, H, W)
        >>> grnd = torch.randint(0, 3, (1, 1, 32, 32))
        >>> # Use softmax=True if input is logits
        >>> loss_func = AsymmetricUnifiedFocalLoss(to_onehot_y=True, use_softmax=True)
        >>> loss = loss_func(pred_logits, grnd)
    """

    def __init__(
        self,
        weight: float = 0.5,
        delta: float = 0.6,
        gamma: float = 0.5,
        include_background: bool = True,
        to_onehot_y: bool = False,
        reduction: LossReduction | str = LossReduction.MEAN,
        use_softmax: bool = False,
        epsilon: float = 1e-7,
    ) -> None:
        """
        Args:
            weight: The weighting factor between Asymmetric Focal Loss and Asymmetric Focal Tversky Loss.
                Final Loss = weight * AFL + (1 - weight) * AFTL. Defaults to 0.5.
            delta: The balancing factor controls the weight of background vs foreground classes.
                Values > 0.5 give more weight to foreground (False Negatives). Defaults to 0.6.
            gamma: The focal exponent. Higher values focus more on hard examples. Defaults to 0.5.
            include_background: If False, channel index 0 (background category) is excluded from the loss calculation.
                Defaults to True.
            to_onehot_y: Whether to convert the label `target` into the one-hot format. Defaults to False.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.
            use_softmax: Whether to use softmax to transform the original logits into probabilities.
                If True, softmax is used. If False, assumes input is already probabilities. Defaults to False.
            epsilon: Small value to prevent division by zero or log(0). Defaults to 1e-7.
        """
        super().__init__(reduction=LossReduction(reduction).value)
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.use_softmax = use_softmax
        self.epsilon = epsilon

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When input and target have incompatible shapes.
        """
        if self.use_softmax:
            input = torch.nn.functional.softmax(input, dim=1)

        n_pred_ch = input.shape[1]

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                if target.shape[1] == 1:
                    target = one_hot(target, num_classes=n_pred_ch)

        if target.shape != input.shape:
            raise ValueError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        # Clip values for numerical stability
        input = torch.clamp(input, self.epsilon, 1.0 - self.epsilon)

        # Part A: Asymmetric Focal Loss
        # Cross Entropy: -target * log(input)
        cross_entropy = -target * torch.log(input)

        # Background (Channel 0): (1 - delta) * (1 - p)^gamma * CE
        back_ce = (1 - self.delta) * torch.pow(1 - input[:, 0:1], self.gamma) * cross_entropy[:, 0:1]

        # Foreground (Channel 1..N): delta * CE
        fore_ce = self.delta * cross_entropy[:, 1:]

        # Combine
        if self.include_background:
            asy_focal_loss = torch.cat([back_ce, fore_ce], dim=1)
        else:
            asy_focal_loss = fore_ce

        # Part B: Asymmetric Focal Tversky Loss
        # Sum over spatial dimensions (Batch and Channel dims are preserved)
        reduce_axis = list(range(2, input.dim()))

        tp = torch.sum(target * input, dim=reduce_axis)
        fn = torch.sum(target * (1 - input), dim=reduce_axis)
        fp = torch.sum((1 - target) * input, dim=reduce_axis)

        # Tversky Index
        dice_class = (tp + self.epsilon) / (tp + self.delta * fn + (1 - self.delta) * fp + self.epsilon)

        # Background: 1 - Dice
        back_dice_loss = 1 - dice_class[:, 0:1]

        # Foreground: (1 - Dice)^(1 - gamma)
        fore_dice_loss = (1 - dice_class[:, 1:]) * torch.pow(1 - dice_class[:, 1:], -self.gamma)

        # Combine
        if self.include_background:
            asy_focal_tversky_loss = torch.cat([back_dice_loss, fore_dice_loss], dim=1)
        else:
            asy_focal_tversky_loss = fore_dice_loss

        # Part C: Unified Combination & Reduction
        # Aggregate Focal Loss spatial dimensions to match Tversky Loss shape (B, C)
        if asy_focal_loss.dim() > 2:
            asy_focal_loss = torch.mean(asy_focal_loss, dim=reduce_axis)

        # Weighted sum
        total_loss = self.weight * asy_focal_loss + (1 - self.weight) * asy_focal_tversky_loss

        if self.reduction == LossReduction.SUM.value:
            return torch.sum(total_loss)
        if self.reduction == LossReduction.NONE.value:
            return total_loss
        if self.reduction == LossReduction.MEAN.value:
            return torch.mean(total_loss)

        raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
