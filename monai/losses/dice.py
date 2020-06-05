# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Callable

import torch
from torch.nn.modules.loss import _Loss

from monai.networks.utils import one_hot


class DiceLoss(_Loss):
    """
    Compute average Dice loss between two tensors. It can support both multi-classes and multi-labels tasks.
    Input logits `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).
    Axis N of `input` is expected to have logit predictions for each class rather than being image channels,
    while the same axis of `target` can be 1 or N (one-hot format). The `smooth` parameter is a value added to the
    intersection and union components of the inter-over-union calculation to smooth results and prevent divide by 0,
    this value should be small. The `include_background` class attribute can be set to False for an instance of
    DiceLoss to exclude the first category (channel index 0) which is by convention assumed to be background.
    If the non-background segmentations are small compared to the total image size they can get overwhelmed by
    the signal from the background so excluding it in such cases helps convergence.
    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
    ):
        """
        Args:
            include_background (bool): If False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y (bool): whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid (bool): If True, apply a sigmoid function to the prediction.
            softmax (bool): If True, apply a softmax function to the prediction.
            squared_pred (bool): use squared versions of targets and predictions in the denominator or not.
            jaccard (bool): compute Jaccard Index (soft IoU) instead of dice or not.
            reduction (`none|mean|sum`): Specifies the reduction to apply to the output:
                ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of elements in the output,
                ``'sum'``: the output will be summed.
                Default: ``'mean'``.
        """
        super().__init__(reduction=reduction)
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        if sigmoid and softmax:
            raise ValueError("sigmoid=True and softmax=True are not compatible.")
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.squared_pred = squared_pred
        self.jaccard = jaccard

    def forward(self, input: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5):
        """
        Args:
            input (tensor): the shape should be BNH[WD].
            target (tensor): the shape should be BNH[WD].
            smooth (float): a small constant to avoid nan.
        """
        if self.sigmoid:
            input = torch.sigmoid(input)
        n_pred_ch = input.shape[1]
        if n_pred_ch == 1:
            if self.softmax:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            if self.to_onehot_y:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            if not self.include_background:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
        else:
            if self.softmax:
                input = torch.softmax(input, 1)
            if self.to_onehot_y:
                target = one_hot(target, n_pred_ch)
            if not self.include_background:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]
        assert (
            target.shape == input.shape
        ), f"ground truth has differing shape ({target.shape}) from input ({input.shape})"

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis = list(range(2, len(input.shape)))
        intersection = torch.sum(target * input, reduce_axis)

        if self.squared_pred:
            target = torch.pow(target, 2)
            input = torch.pow(input, 2)

        ground_o = torch.sum(target, reduce_axis)
        pred_o = torch.sum(input, reduce_axis)

        denominator = ground_o + pred_o

        if self.jaccard:
            denominator -= intersection

        f = 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)
        if self.reduction == "sum":
            return f.sum()  # sum over the batch and channel dims
        if self.reduction == "none":
            return f  # returns [N, n_classes] losses
        if self.reduction == "mean":
            return f.mean()  # the batch and channel average
        raise ValueError(f"reduction={self.reduction} is invalid.")


class GeneralizedDiceLoss(_Loss):
    """
    Compute the generalised Dice loss defined in:

        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017.

    Adapted from:
        https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/layer/loss_segmentation.py#L279
    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        w_type: str = "square",
        reduction: str = "mean",
    ):
        """
        Args:
            include_background (bool): If False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y (bool): whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid (bool): If True, apply a sigmoid function to the prediction.
            softmax (bool): If True, apply a softmax function to the prediction.
            w_type ('square'|'simple'|'uniform'): type of function to transform ground truth volume to a weight factor.
                Default: `'square'`
            reduction (`none|mean|sum`): Specifies the reduction to apply to the output:
                ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the batch size in the output,
                ``'sum'``: the output will be summed over the batch dim.
                Default: ``'mean'``.
        """
        super().__init__(reduction=reduction)
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        if sigmoid and softmax:
            raise ValueError("sigmoid=True and softmax=True are not compatible.")
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.w_func: Callable = torch.ones_like
        if w_type == "simple":
            self.w_func = torch.reciprocal
        elif w_type == "square":
            self.w_func = lambda x: torch.reciprocal(x * x)

    def forward(self, input: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5):
        """
        Args:
            input (tensor): the shape should be BNH[WD].
            target (tensor): the shape should be BNH[WD].
            smooth (float): a small constant to avoid nan.
        """
        if self.sigmoid:
            input = torch.sigmoid(input)
        n_pred_ch = input.shape[1]
        if n_pred_ch == 1:
            if self.softmax:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            if self.to_onehot_y:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            if not self.include_background:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
        else:
            if self.softmax:
                input = torch.softmax(input, 1)
            if self.to_onehot_y:
                target = one_hot(target, n_pred_ch)
            if not self.include_background:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]
        assert (
            target.shape == input.shape
        ), f"ground truth has differing shape ({target.shape}) from input ({input.shape})"

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis = list(range(2, len(input.shape)))
        intersection = torch.sum(target * input, reduce_axis)

        ground_o = torch.sum(target, reduce_axis)
        pred_o = torch.sum(input, reduce_axis)

        denominator = ground_o + pred_o

        w = self.w_func(ground_o.float())
        for b in w:
            infs = torch.isinf(b)
            b[infs] = 0.0
            b[infs] = torch.max(b)

        f = 1.0 - (2.0 * (intersection * w).sum(1) + smooth) / ((denominator * w).sum(1) + smooth)
        if self.reduction == "sum":
            return f.sum()  # sum over the batch dim
        if self.reduction == "none":
            return f  # returns [N] losses
        if self.reduction == "mean":
            return f.mean()  # the batch and channel average
        raise ValueError(f"reduction={self.reduction} is invalid.")


dice = Dice = DiceLoss
generalized_dice = GeneralizedDiceLoss
