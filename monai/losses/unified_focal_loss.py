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
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from monai.networks import one_hot
from monai.utils import LossReduction


class AsymmetricFocalTverskyLoss(_Loss):
    """
    AsymmetricFocalTverskyLoss is a variant of FocalTverskyLoss, which attentions to the foreground class.

    It supports both binary and multi-class segmentation.

    The logic assumes channel 0 is Background, and channels 1..N are Foreground.

    Reimplementation of the Asymmetric Focal Tversky Loss described in:

    - "Unified Focal Loss: Generalising Dice and Cross Entropy-based Losses to Handle Class Imbalanced Medical Image Segmentation",
    Michael Yeung, Computerized Medical Imaging and Graphics
    """

    def __init__(
        self,
        to_onehot_y: bool = False,
        use_softmax: bool = False,
        delta: float = 0.7,
        gamma: float = 0.75,
        epsilon: float = 1e-7,
        reduction: LossReduction | str = LossReduction.MEAN,
    ) -> None:
        """
        Args:
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            use_softmax: whether to use softmax to transform the original logits into probabilities.
                If True, softmax is used. If False, sigmoid is used. Defaults to False.
            delta : weight of the background. Defaults to 0.7.
            gamma : value of the exponent gamma in the definition of the Focal loss  . Defaults to 0.75.
            epsilon : it defines a very small number each time. simmily smooth value. Defaults to 1e-7.
        """
        super().__init__(reduction=LossReduction(reduction).value)
        self.to_onehot_y = to_onehot_y
        self.use_softmax = use_softmax
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.use_softmax:
            y_pred = torch.softmax(y_pred, dim=1)
        else:
            y_pred = torch.sigmoid(y_pred)

        n_pred_ch = y_pred.shape[1]

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                y_true = one_hot(y_true, num_classes=n_pred_ch)

        if y_true.shape != y_pred.shape:
            raise ValueError(f"ground truth has different shape ({y_true.shape}) from input ({y_pred.shape})")

        # clip the prediction to avoid NaN
        y_pred = torch.clamp(y_pred, self.epsilon, 1.0 - self.epsilon)

        spatial_dims = list(range(2, len(y_pred.shape)))

        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = torch.sum(y_true * y_pred, dim=spatial_dims)
        fn = torch.sum(y_true * (1 - y_pred), dim=spatial_dims)
        fp = torch.sum((1 - y_true) * y_pred, dim=spatial_dims)
        dice_class = (tp + self.epsilon) / (tp + self.delta * fn + (1 - self.delta) * fp + self.epsilon)

        # Calculate losses separately for each class, enhancing both classes
        back_dice = 1 - dice_class[:, 0]
        fore_dice = (1 - dice_class[:, 1:]) * torch.pow(1 - dice_class[:, 1:], -self.gamma)

        if fore_dice.shape[1] > 1:
            fore_dice = torch.mean(fore_dice, dim=1)
        else:
            fore_dice = fore_dice.squeeze(1)

        # Average class scores
        loss = torch.mean(torch.stack([back_dice, fore_dice], dim=-1))
        return loss


class AsymmetricFocalLoss(_Loss):
    """
    AsymmetricFocalLoss is a variant of FocalTverskyLoss, which attentions to the foreground class.

    It supports both binary and multi-class segmentation.

    Reimplementation of the Asymmetric Focal Loss described in:

    - "Unified Focal Loss: Generalising Dice and Cross Entropy-based Losses to Handle Class Imbalanced Medical Image Segmentation",
    Michael Yeung, Computerized Medical Imaging and Graphics
    """

    def __init__(
        self,
        to_onehot_y: bool = False,
        use_softmax: bool = False,
        delta: float = 0.7,
        gamma: float = 2,
        epsilon: float = 1e-7,
        reduction: LossReduction | str = LossReduction.MEAN,
    ):
        """
        Args:
            to_onehot_y : whether to convert `y` into the one-hot format. Defaults to False.
            use_softmax: whether to use softmax to transform the original logits into probabilities.
                If True, softmax is used. If False, sigmoid is used. Defaults to False.
            delta : weight of the background. Defaults to 0.7.
            gamma : value of the exponent gamma in the definition of the Focal loss  . Defaults to 0.75.
            epsilon : it defines a very small number each time. simmily smooth value. Defaults to 1e-7.
        """
        super().__init__(reduction=LossReduction(reduction).value)
        self.to_onehot_y = to_onehot_y
        self.use_softmax = use_softmax
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.use_softmax:
            y_log_pred = F.log_softmax(y_pred, dim=1)
            y_pred = torch.exp(y_log_pred)
        else:
            y_log_pred = F.logsigmoid(y_pred)
            y_pred = torch.sigmoid(y_pred)

        n_pred_ch = y_pred.shape[1]

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                y_true = one_hot(y_true, num_classes=n_pred_ch)

        if y_true.shape != y_pred.shape:
            raise ValueError(f"ground truth has different shape ({y_true.shape}) from input ({y_pred.shape})")

        y_pred = torch.clamp(y_pred, self.epsilon, 1.0 - self.epsilon)
        cross_entropy = -y_true * y_log_pred

        back_ce = torch.pow(1 - y_pred[:, 0], self.gamma) * cross_entropy[:, 0]
        back_ce = (1 - self.delta) * back_ce

        fore_ce = cross_entropy[:, 1:]
        fore_ce = self.delta * fore_ce

        if fore_ce.shape[1] > 1:
            fore_ce = torch.sum(fore_ce, dim=1)
        else:
            fore_ce = fore_ce.squeeze(1)

        loss = torch.mean(torch.stack([back_ce, fore_ce], dim=-1))
        return loss


class AsymmetricUnifiedFocalLoss(_Loss):
    """
    AsymmetricUnifiedFocalLoss is a variant of Focal Loss.

    It supports both binary and multi-class segmentation.

    Reimplementation of the Asymmetric Unified Focal Tversky Loss described in:

    - "Unified Focal Loss: Generalising Dice and Cross Entropy-based Losses to Handle Class Imbalanced Medical Image Segmentation",
    Michael Yeung, Computerized Medical Imaging and Graphics
    """

    def __init__(
        self,
        to_onehot_y: bool = False,
        weight: float = 0.5,
        gamma: float = 0.5,
        delta: float = 0.7,
        use_softmax: bool = False,
        reduction: LossReduction | str = LossReduction.MEAN,
    ):
        """
        Args:
            to_onehot_y : whether to convert `y` into the one-hot format. Defaults to False.
            delta : weight of the background. Defaults to 0.7.
            gamma : value of the exponent gamma in the definition of the Focal loss. Defaults to 0.75.
            epsilon : it defines a very small number each time. simmily smooth value. Defaults to 1e-7.
            weight : weight for each loss function, if it's none it's 0.5. Defaults to None.
            use_softmax: whether to use softmax to transform the original logits into probabilities.
                If True, softmax is used. If False, sigmoid is used. Defaults to False.

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
        self.gamma = gamma
        self.delta = delta
        self.weight: float = weight
        self.use_softmax = use_softmax
        self.asy_focal_loss = AsymmetricFocalLoss(
            gamma=self.gamma, delta=self.delta, use_softmax=use_softmax, to_onehot_y=to_onehot_y
        )
        self.asy_focal_tversky_loss = AsymmetricFocalTverskyLoss(
            gamma=self.gamma, delta=self.delta, use_softmax=use_softmax, to_onehot_y=to_onehot_y
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred : the shape should be BNH[WD], where N is the number of classes.
                The input should be the original logits since it will be transformed by
                    a sigmoid in the forward function.
            y_true : the shape should be BNH[WD], where N is the number of classes.

        Raises:
            ValueError: When input and target are different shape
            ValueError: When the number of classes entered does not match the expected number
        """
        if y_pred.shape != y_true.shape:
            raise ValueError(f"ground truth has different shape ({y_true.shape}) from input ({y_pred.shape})")

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
