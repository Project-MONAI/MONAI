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
from typing import Union

import torch
from torch.nn.modules.loss import _Loss

from monai.networks import one_hot
from monai.utils import LossReduction


class TverskyLoss(_Loss):

    """
    Compute the Tversky loss defined in:

        Sadegh et al. (2017) Tversky loss function for image segmentation
        using 3D fully convolutional deep networks. (https://arxiv.org/abs/1706.05721)

    Adapted from:
        https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/layer/loss_segmentation.py#L631

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        alpha: float = 0.5,
        beta: float = 0.5,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
    ):
        """
        Args:
            include_background: If False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: If True, apply a sigmoid function to the prediction.
            softmax: If True, apply a softmax function to the prediction.
            alpha: weight of false positives
            beta: weight of false negatives
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

        Raises:
            ValueError: sigmoid=True and softmax=True are not compatible.

        """

        super().__init__(reduction=LossReduction(reduction))
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y

        if sigmoid and softmax:
            raise ValueError("sigmoid=True and softmax=True are not compatible.")
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.alpha = alpha
        self.beta = beta

    def forward(self, input: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5):
        """
        Args:
            input (tensor): the shape should be BNH[WD].
            target (tensor): the shape should be BNH[WD].
            smooth: a small constant to avoid nan.

        Raises:
            ValueError: reduction={self.reduction} is invalid.

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

        p0 = input
        p1 = 1 - p0
        g0 = target
        g1 = 1 - g0

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis = list(range(2, len(input.shape)))

        tp = torch.sum(p0 * g0, reduce_axis)
        fp = self.alpha * torch.sum(p0 * g1, reduce_axis)
        fn = self.beta * torch.sum(p1 * g0, reduce_axis)

        numerator = tp + smooth
        denominator = tp + fp + fn + smooth

        score = 1.0 - numerator / denominator

        if self.reduction == LossReduction.SUM:
            return score.sum()  # sum over the batch and channel dims
        if self.reduction == LossReduction.NONE:
            return score  # returns [N, n_classes] losses
        if self.reduction == LossReduction.MEAN:
            return score.mean()
        raise ValueError(f"reduction={self.reduction} is invalid.")
