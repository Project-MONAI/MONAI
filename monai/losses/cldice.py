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
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from monai.losses.dice import DiceLoss
from monai.networks import one_hot
from monai.utils import LossReduction
from monai.utils.deprecate_utils import deprecated_arg


def soft_erode(img: torch.Tensor) -> torch.Tensor:  # type: ignore
    """
    Perform soft erosion on the input image

    Args:
        img: the shape should be BCH(WD)

    Adapted from:
        https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/soft_skeleton.py#L6
    """
    if len(img.shape) == 4:
        p1 = -(F.max_pool2d(-img, (3, 1), (1, 1), (1, 0)))
        p2 = -(F.max_pool2d(-img, (1, 3), (1, 1), (0, 1)))
        return torch.min(p1, p2)
    elif len(img.shape) == 5:
        p1 = -(F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0)))
        p2 = -(F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0)))
        p3 = -(F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1)))
        return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:  # type: ignore
    """
    Perform soft dilation on the input image

    Args:
        img: the shape should be BCH(WD)

    Adapted from:
        https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/soft_skeleton.py#L18
    """
    if len(img.shape) == 4:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
    elif len(img.shape) == 5:
        return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def soft_open(img: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to perform soft opening on the input image

    Args:
        img: the shape should be BCH(WD)

    Adapted from:
        https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/soft_skeleton.py#L25
    """
    eroded_image = soft_erode(img)
    dilated_image = soft_dilate(eroded_image)
    return dilated_image


def soft_skel(img: torch.Tensor, iter_: int) -> torch.Tensor:
    """
    Perform soft skeletonization on the input image

    Adapted from:
       https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/soft_skeleton.py#L29

    Args:
        img: the shape should be BCH(WD)
        iter_: number of iterations for skeletonization

    Returns:
        skeletonized image
    """
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


class SoftclDiceLoss(_Loss):
    """
    Compute the Soft clDice loss defined in:

        Shit et al. (2021) clDice -- A Novel Topology-Preserving Loss Function
        for Tubular Structure Segmentation. (https://arxiv.org/abs/2003.07311)

    Adapted from:
        https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/cldice.py#L7

    The data `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).
    Note that axis N of `input` is expected to be logits or probabilities for each class, if passing logits as input,
    must set `sigmoid=True` or `softmax=True`, or specifying `other_act`. And the same axis of `target`
    can be 1 or N (one-hot format).

    """

    def __init__(
        self,
        iter_: int = 3,
        smooth_nr: float = 1.0,
        smooth_dr: float = 1.0,
        smooth: float = 1e-4,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        reduction: LossReduction | str = LossReduction.MEAN,
    ) -> None:
        """
        Args:
            iter_: Number of iterations for skeletonization. Must be a non-negative integer. Defaults to 3.
            smooth_nr: a small constant added to the numerator to avoid zero. Defaults to 1.0.
            smooth_dr: a small constant added to the denominator of the individual precision /
                sensitivity ratios and the internal Dice denominator to avoid nan. Defaults to 1.0.
            smooth: a small constant added to the denominator of the harmonic mean to avoid nan. Defaults to 1e-4.
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
                if the non-background segmentations are small compared to the total image size they can get overwhelmed
                by the signal from the background so excluding it in such cases helps convergence.
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

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            TypeError: When ``iter_`` is not an ``int``.
            ValueError: When ``iter_`` is a negative integer.
            ValueError: When ``smooth`` is not a positive value.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        if not isinstance(iter_, int):
            raise TypeError(f"iter_ must be an integer but got {type(iter_).__name__}.")
        if iter_ < 0:
            raise ValueError(f"iter_ must be a non-negative integer but got {iter_}.")
        if smooth <= 0:
            raise ValueError(f"smooth must be a positive value but got {smooth}.")
        self.iter = iter_
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.smooth = float(smooth)
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act

    @deprecated_arg("y_pred", since="1.5", removed="1.8", new_name="input", msg_suffix="please use `input` instead.")
    @deprecated_arg("y_true", since="1.5", removed="1.8", new_name="target", msg_suffix="please use `target` instead.")
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.

        """
        n_pred_ch = input.shape[1]

        if self.sigmoid:
            input = torch.sigmoid(input)

        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.", stacklevel=2)
            else:
                input = torch.softmax(input, dim=1)

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
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        skel_pred = soft_skel(input, self.iter)
        skel_true = soft_skel(target, self.iter)

        # Compute per-batch clDice by reducing over channel and spatial dimensions
        # reduce_axis includes all dimensions except batch (dim 0)
        reduce_axis: list[int] = list(range(1, len(input.shape)))

        tprec = (torch.sum(torch.multiply(skel_pred, target), dim=reduce_axis) + self.smooth_nr) / (
            torch.sum(skel_pred, dim=reduce_axis) + self.smooth_dr
        )
        tsens = (torch.sum(torch.multiply(skel_true, input), dim=reduce_axis) + self.smooth_nr) / (
            torch.sum(skel_true, dim=reduce_axis) + self.smooth_dr
        )
        # Add small epsilon for numerical stability in harmonic mean
        cl_dice: torch.Tensor = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens + self.smooth)

        # Apply reduction
        if self.reduction == LossReduction.MEAN.value:
            cl_dice = torch.mean(cl_dice)
        elif self.reduction == LossReduction.SUM.value:
            cl_dice = torch.sum(cl_dice)
        elif self.reduction == LossReduction.NONE.value:
            pass  # keep per-batch values
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return cl_dice


class SoftDiceclDiceLoss(_Loss):
    """
    Compute both Dice loss and clDice loss, and return the weighted sum of these two losses.
    The details of Dice loss is shown in ``monai.losses.DiceLoss``.
    The details of clDice loss is shown in ``monai.losses.SoftclDiceLoss``.

    Adapted from:
        Shit et al. (2021) clDice -- A Novel Topology-Preserving Loss Function
        for Tubular Structure Segmentation. (https://arxiv.org/abs/2003.07311)

    """

    def __init__(
        self,
        iter_: int = 3,
        alpha: float = 0.5,
        smooth_nr: float = 1.0,
        smooth_dr: float = 1.0,
        smooth: float = 1e-4,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        reduction: LossReduction | str = LossReduction.MEAN,
    ) -> None:
        """
        Args:
            iter_: Number of iterations for skeletonization, used by clDice. Must be a non-negative integer. Defaults to 3.
            alpha: Weighing factor for cldice component. Total loss = (1 - alpha) * dice + alpha * cldice.
                Defaults to 0.5.
            smooth_nr: a small constant added to the numerator to avoid zero, used by both Dice and clDice. Defaults to 1.0.
            smooth_dr: a small constant added to the denominator to avoid nan, used by both Dice and clDice. Defaults to 1.0.
            smooth: a small constant added to the denominator of the harmonic mean in clDice to avoid nan.
                Defaults to 1e-4. Note: This differs from standalone DiceLoss defaults (1e-5) to follow clDice convention.
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
                if the non-background segmentations are small compared to the total image size they can get overwhelmed
                by the signal from the background so excluding it in such cases helps convergence.
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

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When ``alpha`` is not in ``[0, 1]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1] but got {alpha}.")
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=False,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
        )
        self.cldice = SoftclDiceLoss(
            iter_=iter_,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            smooth=smooth,
            include_background=include_background,
            to_onehot_y=False,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            reduction=reduction,
        )
        self.alpha = alpha
        self.to_onehot_y = to_onehot_y

    @deprecated_arg("y_pred", since="1.5", removed="1.8", new_name="input", msg_suffix="please use `input` instead.")
    @deprecated_arg("y_true", since="1.5", removed="1.8", new_name="target", msg_suffix="please use `target` instead.")
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        if input.dim() != target.dim():
            raise ValueError(
                f"the number of dimensions for input and target should be the same, got shape {input.shape} and {target.shape}."
            )

        if target.shape[1] != 1 and target.shape[1] != input.shape[1]:
            raise ValueError(
                f"number of channels for target is neither 1 nor the same as input, got shape {input.shape} and {target.shape}."
            )

        if self.to_onehot_y:
            n_pred_ch = input.shape[1]
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.", stacklevel=2)
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        dice_loss = self.dice(input, target)
        cldice_loss = self.cldice(input, target)
        total_loss: torch.Tensor = (1.0 - self.alpha) * dice_loss + self.alpha * cldice_loss

        return total_loss
