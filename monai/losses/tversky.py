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

import torch
from torch.nn.modules.loss import _Loss

from monai.networks.utils import one_hot


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
        do_sigmoid: bool = False,
        do_softmax: bool = False,
        alpha: float = 0.5,
        beta: float = 0.5,
        reduction: str = "mean",
    ):

        """
        Args:
            include_background (bool): If False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y (bool): whether to convert `y` into the one-hot format. Defaults to False.
            do_sigmoid (bool): If True, apply a sigmoid function to the prediction.
            do_softmax (bool): If True, apply a softmax function to the prediction.
            alpha (float): weight of false positives
            beta  (float): weight of false negatives
            reduction (`none|mean|sum`): Specifies the reduction to apply to the output:
                ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of elements in the output,
                ``'sum'``: the output will be summed.
                Default: ``'mean'``.

        """

        super().__init__(reduction=reduction)
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y

        if do_sigmoid and do_softmax:
            raise ValueError("do_sigmoid=True and do_softmax=True are not compatible.")
        self.do_sigmoid = do_sigmoid
        self.do_softmax = do_softmax
        self.alpha = alpha
        self.beta = beta

    def forward(self, input: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5):
        """
        Args:
            input (tensor): the shape should be BNH[WD].
            target (tensor): the shape should be BNH[WD].
            smooth (float): a small constant to avoid nan.
        """
        if self.do_sigmoid:
            input = torch.sigmoid(input)
        n_pred_ch = input.shape[1]
        if n_pred_ch == 1:
            if self.do_softmax:
                warnings.warn("single channel prediction, `do_softmax=True` ignored.")
            if self.to_onehot_y:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            if not self.include_background:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
        else:
            if self.do_softmax:
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

        if self.reduction == "sum":
            return score.sum()  # sum over the batch and channel dims
        if self.reduction == "none":
            return score  # returns [N, n_classes] losses
        if self.reduction == "mean":
            return score.mean()
        raise ValueError(f"reduction={self.reduction} is invalid.")
