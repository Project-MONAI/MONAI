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


class AsymmetricFocalTverskyLoss(_Loss):
    """
    AsymmetricFocalTverskyLoss is a variant of FocalTverskyLoss, which attentions to the foreground class.

    Actually, it's only supported for binary image segmentation now.

    Reimplementation of the Asymmetric Focal Tversky Loss described in:

    - "Unified Focal Loss: Generalising Dice and Cross Entropy-based Losses to Handle Class Imbalanced Medical Image Segmentation",
    Michael Yeung, Computerized Medical Imaging and Graphics
    """

    def __init__(
        self,
        to_onehot_y: bool = False,
        delta: float = 0.7,
        gamma: float = 0.75,
        epsilon: float = 1e-7,
        reduction: LossReduction | str = LossReduction.MEAN,
        ignore_index: int | None = None,
    ) -> None:
        """
        Args:
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            delta : weight of the background. Defaults to 0.7.
            gamma : value of the exponent gamma in the definition of the Focal loss. Defaults to 0.75.
            epsilon : it defines a very small number each time. similarly smooth value. Defaults to 1e-7.
            ignore_index: class index to ignore from the loss computation.
        """
        super().__init__(reduction=LossReduction(reduction).value)
        self.to_onehot_y = to_onehot_y
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        n_pred_ch = y_pred.shape[1]

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                y_true = one_hot(y_true, num_classes=n_pred_ch)

        if y_true.shape != y_pred.shape:
            raise ValueError(f"ground truth has different shape ({y_true.shape}) from input ({y_pred.shape})")

        # Handle ignore_index:
        # Since the wrapper already zeroed out ignore_index in y_true,
        # we create the mask where y_true has any valid class (sum across channels > 0)
        mask = torch.ones_like(y_true)
        if self.ignore_index is not None:
            # We identify valid pixels: where at least one channel is 1
            spatial_mask = (torch.sum(y_true, dim=1, keepdim=True) > 0).float()
            mask = spatial_mask.expand_as(y_true)
            # Ensure y_pred is also masked so it doesn't create False Positives in ignored areas
            y_pred = y_pred * mask

        # clip the prediction to avoid NaN
        y_pred = torch.clamp(y_pred, self.epsilon, 1.0 - self.epsilon)
        axis = list(range(2, len(y_pred.shape)))

        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = torch.sum(y_true * y_pred, dim=axis)
        fn = torch.sum(y_true * (1 - y_pred) * mask, dim=axis)
        fp = torch.sum((1 - y_true) * y_pred * mask, dim=axis)
        dice_class = (tp + self.epsilon) / (tp + self.delta * fn + (1 - self.delta) * fp + self.epsilon)

        # Calculate losses separately for each class, enhancing both classes
        back_dice = 1 - dice_class[:, 0]
        fore_dice = (1 - dice_class[:, 1]) * torch.pow(1 - dice_class[:, 1], -self.gamma)

        # Average class scores
        loss = torch.stack([back_dice, fore_dice], dim=-1)
        if self.reduction == LossReduction.MEAN.value:
            return torch.mean(loss)
        if self.reduction == LossReduction.SUM.value:
            return torch.sum(loss)
        return loss


class AsymmetricFocalLoss(_Loss):
    """
    AsymmetricFocalLoss is a variant of FocalTverskyLoss, which attentions to the foreground class.

    Actually, it's only supported for binary image segmentation now.

    Reimplementation of the Asymmetric Focal Loss described in:

    - "Unified Focal Loss: Generalising Dice and Cross Entropy-based Losses to Handle Class Imbalanced Medical Image Segmentation",
    Michael Yeung, Computerized Medical Imaging and Graphics
    """

    def __init__(
        self,
        to_onehot_y: bool = False,
        delta: float = 0.7,
        gamma: float = 2,
        epsilon: float = 1e-7,
        reduction: LossReduction | str = LossReduction.MEAN,
        ignore_index: int | None = None,
    ):
        """
        Args:
            to_onehot_y : whether to convert `y` into the one-hot format. Defaults to False.
            delta : weight of the background. Defaults to 0.7.
            gamma : value of the exponent gamma in the definition of the Focal loss. Defaults to 2.
            epsilon : it defines a very small number each time. similarly smooth value. Defaults to 1e-7.
            ignore_index: class index to ignore from the loss computation.
        """
        super().__init__(reduction=LossReduction(reduction).value)
        self.to_onehot_y = to_onehot_y
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        n_pred_ch = y_pred.shape[1]

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                y_true = one_hot(y_true, num_classes=n_pred_ch)

        if y_true.shape != y_pred.shape:
            raise ValueError(f"ground truth has different shape ({y_true.shape}) from input ({y_pred.shape})")

        y_pred = torch.clamp(y_pred, self.epsilon, 1.0 - self.epsilon)
        cross_entropy = -y_true * torch.log(y_pred)

        if self.ignore_index is not None:
            spatial_mask = (torch.sum(y_true, dim=1, keepdim=True) > 0).float()
            cross_entropy = cross_entropy * spatial_mask

        back_ce = torch.pow(1 - y_pred[:, 0], self.gamma) * cross_entropy[:, 0]
        back_ce = (1 - self.delta) * back_ce

        fore_ce = cross_entropy[:, 1]
        fore_ce = self.delta * fore_ce

        loss = torch.stack([back_ce, fore_ce], dim=1)  # [B, 2, H, W]
        if self.reduction == LossReduction.MEAN.value:
            if self.ignore_index is not None:
                # Normalize by the number of non-ignored pixels
                return loss.sum() / spatial_mask.sum().clamp(min=1e-5)
            return loss.mean()
        if self.reduction == LossReduction.SUM.value:
            return loss.sum()
        return loss


class AsymmetricUnifiedFocalLoss(_Loss):
    """
    AsymmetricUnifiedFocalLoss is a variant of Focal Loss.

    Actually, it's only supported for binary image segmentation now

    Reimplementation of the Asymmetric Unified Focal Tversky Loss described in:

    - "Unified Focal Loss: Generalising Dice and Cross Entropy-based Losses to Handle Class Imbalanced Medical Image Segmentation",
    Michael Yeung, Computerized Medical Imaging and Graphics
    """

    def __init__(
        self,
        to_onehot_y: bool = False,
        num_classes: int = 2,
        weight: float = 0.5,
        gamma: float = 0.5,
        delta: float = 0.7,
        reduction: LossReduction | str = LossReduction.MEAN,
        ignore_index: int | None = None,
    ):
        """
        Args:
            to_onehot_y : whether to convert `y` into the one-hot format. Defaults to False.
            num_classes : number of classes, it only supports 2 now. Defaults to 2.
            weight : weight for each loss function. Defaults to 0.5.
            gamma : value of the exponent gamma in the definition of the Focal loss. Defaults to 0.5.
            delta : weight of the background. Defaults to 0.7.
            ignore_index: class index to ignore from the loss computation.

        Example:
            >>> import torch
            >>> from monai.losses import AsymmetricUnifiedFocalLoss
            >>> pred = torch.ones((1,1,32,32), dtype=torch.float32)
            >>> grnd = torch.ones((1,1,32,32), dtype=torch.int64)
            >>> fl = AsymmetricUnifiedFocalLoss(to_onehot_y=True)
            >>> fl(pred, grnd)
        """
        super().__init__(reduction=LossReduction(reduction).value)
        self.to_onehot_y = to_onehot_y
        self.num_classes = num_classes
        self.gamma = gamma
        self.delta = delta
        self.weight: float = weight
        self.asy_focal_loss = AsymmetricFocalLoss(gamma=self.gamma, delta=self.delta, ignore_index=ignore_index)
        self.asy_focal_tversky_loss = AsymmetricFocalTverskyLoss(
            gamma=self.gamma, delta=self.delta, ignore_index=ignore_index
        )
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred : the shape should be BNH[WD], where N is the number of classes.
                It only supports binary segmentation.
                The input should be the original logits since it will be transformed by
                    a sigmoid in the forward function.
            y_true : the shape should be BNH[WD], where N is the number of classes.
                It only supports binary segmentation.

        Raises:
            ValueError: When input and target are different shape
            ValueError: When len(y_pred.shape) != 4 and len(y_pred.shape) != 5
            ValueError: When num_classes
            ValueError: When the number of classes entered does not match the expected number
        """
        if len(y_pred.shape) != 4 and len(y_pred.shape) != 5:
            raise ValueError(f"input shape must be 4 or 5, but got {y_pred.shape}")

        # 2. Transform binary inputs to 2-channel space
        if y_pred.shape[1] == 1:
            y_pred = torch.cat([-y_pred, y_pred], dim=1)

            if self.ignore_index is not None:
                mask = (y_true != self.ignore_index).float()
                y_true_clean = torch.where(y_true == self.ignore_index, 0, y_true)
                y_true = one_hot(y_true_clean, num_classes=self.num_classes)
                y_true = y_true * mask
            else:
                y_true = one_hot(y_true, num_classes=self.num_classes)

        # 3. NOW check if shapes match (They should both be [B, 2, H, W] now)
        if y_true.shape != y_pred.shape:
            raise ValueError(f"ground truth has different shape ({y_true.shape}) from input ({y_pred.shape})")

        if torch.max(y_true) != self.num_classes - 1:
            raise ValueError(f"Please make sure the number of classes is {self.num_classes - 1}")

        asy_focal_loss = self.asy_focal_loss(y_pred, y_true)
        asy_focal_tversky_loss = self.asy_focal_tversky_loss(y_pred, y_true)

        loss: torch.Tensor = self.weight * asy_focal_loss + (1 - self.weight) * asy_focal_tversky_loss

        if self.reduction == LossReduction.SUM.value:
            return torch.sum(loss)  # sum over the batch and channel dims
        if self.reduction == LossReduction.NONE.value:
            return loss  # returns [N, num_classes] losses
        if self.reduction == LossReduction.MEAN.value:
            return torch.mean(loss)
        raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
