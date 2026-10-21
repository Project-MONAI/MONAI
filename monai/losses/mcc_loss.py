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
from collections.abc import Callable

import torch
from torch.nn.modules.loss import _Loss

from monai.networks import one_hot
from monai.utils import LossReduction


class MCCLoss(_Loss):
    """
    Compute the Matthews Correlation Coefficient (MCC) loss between two tensors.

    Unlike Dice and Tversky losses which only use TP, FP, and FN, the MCC loss considers all four entries
    of the confusion matrix (TP, TN, FP, FN), making it effective for class-imbalanced segmentation tasks
    where background dominates the image. The loss is computed as ``1 - MCC`` where
    ``MCC = (TP * TN - FP * FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))``.

    The soft confusion matrix entries are computed as:

        - ``TP = sum(input * target)``
        - ``TN = sum((1 - input) * (1 - target))``
        - ``FP = sum(input * (1 - target))``
        - ``FN = sum((1 - input) * target)``

    The data `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).

    Note that axis N of `input` is expected to be logits or probabilities for each class, if passing logits as input,
    must set `sigmoid=True` or `softmax=True`, or specifying `other_act`. And the same axis of `target`
    can be 1 or N (one-hot format).

    The original paper:

        Abhishek, K. and Hamarneh, G. (2021) Matthews Correlation Coefficient Loss for Deep Convolutional
        Networks: Application to Skin Lesion Segmentation. IEEE ISBI, pp. 225-229.
        (https://doi.org/10.1109/ISBI48211.2021.9433782)

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        reduction: LossReduction | str = LossReduction.MEAN,
        smooth_nr: float = 0.0,
        smooth_dr: float = 1e-5,
        batch: bool = False,
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
                if the non-background segmentations are small compared to the total image size they can get
                overwhelmed by the signal from the background so excluding it in such cases helps convergence.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: callable function to execute other activation layers, Defaults to ``None``. for example:
                ``other_act = torch.tanh``.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the confusion matrix entries over the batch dimension before computing MCC.
                Defaults to False, MCC is computed independently for each item in the batch
                before any `reduction`.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        Example:
            >>> from monai.losses.mcc_loss import MCCLoss
            >>> import torch
            >>> B, C, H, W = 7, 1, 3, 2
            >>> input = torch.rand(B, C, H, W)
            >>> target = torch.randint(low=0, high=2, size=(B, C, H, W)).float()
            >>> self = MCCLoss(reduction='none')
            >>> loss = self(input, target)
        """
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.", stacklevel=2)
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.", stacklevel=2)
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.", stacklevel=2)
            else:
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has differing shape ({target.shape}) from input ({input.shape})")

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: list[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            reduce_axis = [0] + reduce_axis

        # Soft confusion matrix entries (Eq. 5 in the paper).
        tp = torch.sum(input * target, dim=reduce_axis)
        tn = torch.sum((1.0 - input) * (1.0 - target), dim=reduce_axis)
        fp = torch.sum(input * (1.0 - target), dim=reduce_axis)
        fn = torch.sum((1.0 - input) * target, dim=reduce_axis)

        # MCC (Eq. 3) and loss (Eq. 4).
        numerator = tp * tn - fp * fn + self.smooth_nr
        denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + self.smooth_dr)

        mcc = numerator / denominator
        score: torch.Tensor = 1.0 - mcc

        # When fp = fn = 0, prediction is perfect but the denominator product
        # tends to 0 when tp = 0 or tn = 0, giving mcc ~ 0 instead of 1.
        perfect = (fp == 0) & (fn == 0)
        score = torch.where(perfect, torch.zeros_like(score), score)

        if self.reduction == LossReduction.SUM.value:
            return torch.sum(score)
        if self.reduction == LossReduction.NONE.value:
            return score
        if self.reduction == LossReduction.MEAN.value:
            return torch.mean(score)
        raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
