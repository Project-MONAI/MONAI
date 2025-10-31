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
    It treats the background class (index 0) differently from all foreground classes (indices 1...N).

    Reimplementation of the Asymmetric Focal Tversky Loss described in:

    - "Unified Focal Loss: Generalising Dice and Cross Entropy-based Losses to Handle Class Imbalanced Medical Image Segmentation",
      Michael Yeung, Computerized Medical Imaging and Graphics
    """

    def __init__(
        self,
        include_background: bool = True,
        delta: float = 0.7,
        gamma: float = 0.75,
        epsilon: float = 1e-7,
        reduction: LossReduction | str = LossReduction.MEAN,
    ) -> None:
        """
        Args:
            include_background: whether to include loss computation for the background class. Defaults to True.
            delta : weight of the background. Defaults to 0.7. (Used to weigh FNs and FPs in Tversky index)
            gamma : value of the exponent gamma in the definition of the Focal loss. Defaults to 0.75.
            epsilon : it defines a very small number each time. similarly smooth value. Defaults to 1e-7.
            reduction: specifies the reduction to apply to the output: "none", "mean", "sum".
        """
        super().__init__(reduction=LossReduction(reduction).value)
        self.include_background = include_background
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        n_pred_ch = y_pred.shape[1]

        if y_true.shape != y_pred.shape:
            raise ValueError(f"ground truth has different shape ({y_true.shape}) from input ({y_pred.shape})")

        # Exclude background if needed
        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                y_pred = y_pred[:, 1:]
                y_true = y_true[:, 1:]

        axis = list(range(2, len(y_pred.shape)))

        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = torch.sum(y_true * y_pred, dim=axis)
        fn = torch.sum(y_true * (1 - y_pred), dim=axis)
        fp = torch.sum((1 - y_true) * y_pred, dim=axis)
        dice_class = (tp + self.epsilon) / (tp + self.delta * fn + (1 - self.delta) * fp + self.epsilon)
        # dice_class shape is (B, C)

        n_classes = dice_class.shape[1]

        if not self.include_background:
            # All classes are foreground, apply foreground logic
            loss = torch.pow(1.0 - dice_class, 1.0 - self.gamma)  # (B, C)
        elif n_classes == 1:
            # Single class, must be foreground (BG was excluded or not provided)
            loss = torch.pow(1.0 - dice_class, 1.0 - self.gamma)  # (B, 1)
        else:
            # Asymmetric logic: class 0 is BG, others are FG
            back_dice_loss = (1.0 - dice_class[:, 0]).unsqueeze(1)  # (B, 1)
            fore_dice_loss = torch.pow(1.0 - dice_class[:, 1:], 1.0 - self.gamma)  # (B, C-1)
            loss = torch.cat([back_dice_loss, fore_dice_loss], dim=1)  # (B, C)

        # Apply reduction
        if self.reduction == LossReduction.MEAN.value:
            return torch.mean(loss)  # mean over batch and classes
        if self.reduction == LossReduction.SUM.value:
            return torch.sum(loss)  # sum over batch and classes
        if self.reduction == LossReduction.NONE.value:
            return loss  # returns (B, C) losses
        raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')


class AsymmetricFocalLoss(_Loss):
    """
    AsymmetricFocalLoss is a variant of FocalLoss, which attentions to the foreground class.
    It treats the background class (index 0) differently from all foreground classes (indices 1...N).
    Background class (0): applies gamma exponent to (1-p)
    Foreground classes (1..N): no gamma exponent

    Reimplementation of the Asymmetric Focal Loss described in:

    - "Unified Focal Loss: Generalising Dice and Cross Entropy-based Losses to Handle Class Imbalanced Medical Image Segmentation",
      Michael Yeung, Computerized Medical Imaging and Graphics
    """

    def __init__(
        self,
        include_background: bool = True,
        delta: float = 0.7,
        gamma: float = 2.0,
        epsilon: float = 1e-7,
        reduction: LossReduction | str = LossReduction.MEAN,
    ):
        """
        Args:
            include_background: whether to include loss computation for the background class. Defaults to True.
            delta : weight of the foreground. Defaults to 0.7. (1-delta is weight of background)
            gamma : value of the exponent gamma in the definition of the Focal loss. Defaults to 2.0.
            epsilon : it defines a very small number each time. similarly smooth value. Defaults to 1e-7.
            reduction: specifies the reduction to apply to the output: "none", "mean", "sum".
        """
        super().__init__(reduction=LossReduction(reduction).value)
        self.include_background = include_background
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        n_pred_ch = y_pred.shape[1]

        if y_true.shape != y_pred.shape:
            raise ValueError(f"ground truth has different shape ({y_true.shape}) from input ({y_pred.shape})")

        # Exclude background if needed
        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                y_pred = y_pred[:, 1:]
                y_true = y_true[:, 1:]

        y_pred = torch.clamp(y_pred, self.epsilon, 1.0 - self.epsilon)
        cross_entropy = -y_true * torch.log(y_pred)  # Shape (B, C, H, W, [D])

        n_classes = y_pred.shape[1]

        if not self.include_background:
            # All classes are foreground, apply foreground logic
            loss = self.delta * cross_entropy  # (B, C, H, W)
        elif n_classes == 1:
            # Single class, must be foreground
            loss = self.delta * cross_entropy  # (B, 1, H, W)
        else:
            # Asymmetric logic: class 0 is BG, others are FG
            # (B, H, W)
            back_ce = (1.0 - self.delta) * torch.pow(1.0 - y_pred[:, 0], self.gamma) * cross_entropy[:, 0]
            # (B, C-1, H, W)
            fore_ce = self.delta * cross_entropy[:, 1:]

            loss = torch.cat([back_ce.unsqueeze(1), fore_ce], dim=1)  # (B, C, H, W)

        # Apply reduction
        if self.reduction == LossReduction.MEAN.value:
            return torch.mean(loss)  # mean over batch, class, and spatial
        if self.reduction == LossReduction.SUM.value:
            return torch.sum(loss)
        if self.reduction == LossReduction.NONE.value:
            return loss  # returns (B, C, H, W)
        raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')


class AsymmetricUnifiedFocalLoss(_Loss):
    """
    AsymmetricUnifiedFocalLoss is a variant of Focal Loss, combining AsymmetricFocalLoss
    and AsymmetricFocalTverskyLoss.

    Reimplementation of the Asymmetric Unified Focal Tversky Loss described in:

    - "Unified Focal Loss: Generalising Dice and Cross Entropy-based Losses to Handle Class Imbalanced Medical Image Segmentation",
      Michael Yeung, Computerized Medical Imaging and Graphics
    """

    def __init__(
        self,
        to_onehot_y: bool = False,
        use_sigmoid: bool = False,
        use_softmax: bool = False,
        lambda_focal: float = 0.5,
        focal_loss_gamma: float = 2.0,
        focal_loss_delta: float = 0.7,
        tversky_loss_gamma: float = 0.75,
        tversky_loss_delta: float = 0.7,
        include_background: bool = True,
        reduction: LossReduction | str = LossReduction.MEAN,
    ):
        """
        Args:
            to_onehot_y : whether to convert `y` into the one-hot format. Defaults to False.
            use_sigmoid: if True, apply a sigmoid activation to the input y_pred.
            use_softmax: if True, apply a softmax activation to the input y_pred.
            lambda_focal: the weight for AsymmetricFocalLoss (Cross-Entropy based).
                The weight for AsymmetricFocalTverskyLoss will be (1 - lambda_focal). Defaults to 0.5.
            focal_loss_gamma: gamma parameter for the AsymmetricFocalLoss component. Defaults to 2.0.
            focal_loss_delta: delta parameter for the AsymmetricFocalLoss component. Defaults to 0.7.
            tversky_loss_gamma: gamma parameter for the AsymmetricFocalTverskyLoss component. Defaults to 0.75.
            tversky_loss_delta: delta parameter for the AsymmetricFocalTverskyLoss component. Defaults to 0.7.
            include_background: whether to include loss computation for the background class. Defaults to True.
            reduction: specifies the reduction to apply to the output: "none", "mean", "sum".

        Example:
            >>> import torch
            >>> from monai.losses import AsymmetricUnifiedFocalLoss
            >>> pred = torch.randn((1, 2, 32, 32), dtype=torch.float32)
            >>> grnd = torch.randint(0, 2, (1, 1, 32, 32), dtype=torch.int64)
            >>> fl = AsymmetricUnifiedFocalLoss(use_softmax=True, to_onehot_y=True)
            >>> fl(pred, grnd)
        """
        super().__init__(reduction=LossReduction(reduction).value)
        self.to_onehot_y = to_onehot_y
        self.use_sigmoid = use_sigmoid
        self.use_softmax = use_softmax
        self.lambda_focal = lambda_focal
        self.include_background = include_background

        if self.use_sigmoid and self.use_softmax:
            raise ValueError("Both use_sigmoid and use_softmax cannot be True.")

        self.asy_focal_loss = AsymmetricFocalLoss(
            include_background=self.include_background,
            gamma=focal_loss_gamma,
            delta=focal_loss_delta,
            reduction=self.reduction,
        )
        self.asy_focal_tversky_loss = AsymmetricFocalTverskyLoss(
            include_background=self.include_background,
            gamma=tversky_loss_gamma,
            delta=tversky_loss_delta,
            reduction=self.reduction,
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred : the shape should be BNH[WD].
            y_true : the shape should be BNH[WD] or B1H[WD].
        """
        n_pred_ch = y_pred.shape[1]

        y_pred_act = y_pred
        if self.use_sigmoid:
            y_pred_act = torch.sigmoid(y_pred)
        elif self.use_softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, use_softmax=True ignored.")
            else:
                y_pred_act = torch.softmax(y_pred, dim=1)

        if self.to_onehot_y:
            if n_pred_ch == 1 and not self.use_sigmoid:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            elif n_pred_ch > 1 or self.use_sigmoid:
                # Ensure y_true is (B, 1, H, W, [D]) for one-hot conversion
                if y_true.shape[1] != 1:
                    y_true = y_true.unsqueeze(1)
                y_true = one_hot(y_true, num_classes=n_pred_ch)

        # Ensure y_true has the same shape as y_pred_act
        if y_true.shape != y_pred_act.shape:
            # This can happen if y_true is (B, H, W) and y_pred is (B, 1, H, W) after sigmoid
            if y_true.shape[1] != y_pred_act.shape[1] and y_true.ndim == y_pred_act.ndim - 1:
                y_true = y_true.unsqueeze(1)  # Add channel dim

            if y_true.shape != y_pred_act.shape:
                raise ValueError(
                    f"ground truth has different shape ({y_true.shape}) from input ({y_pred_act.shape}) "
                    f"after activations/one-hot"
                )

        f_loss = self.asy_focal_loss(y_pred_act, y_true)
        t_loss = self.asy_focal_tversky_loss(y_pred_act, y_true)

        loss: torch.Tensor = self.lambda_focal * f_loss + (1 - self.lambda_focal) * t_loss

        return loss
