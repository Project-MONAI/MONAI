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

from typing import Union
import warnings

import torch
from monai.networks import one_hot
from monai.utils import MetricReduction


class DiceMetric:
    """
    Compute average Dice loss between two tensors. It can support both multi-classes and multi-labels tasks.
    Input logits `y_pred` (BNHW[D] where N is number of classes) is compared with ground truth `y` (BNHW[D]).
    Axis N of `y_preds` is expected to have logit predictions for each class rather than being image channels,
    while the same axis of `y` can be 1 or N (one-hot format). The `include_background` class attribute can be
    set to False for an instance of DiceLoss to exclude the first category (channel index 0) which is by
    convention assumed to be background. If the non-background segmentations are small compared to the total
    image size they can get overwhelmed by the signal from the background so excluding it in such cases helps
    convergence.

    Args:
        include_background: whether to skip Dice computation on the first channel of
            the predicted output. Defaults to True.
        to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
        mutually_exclusive: if True, `y_pred` will be converted into a binary matrix using
            a combination of argmax and to_onehot.  Defaults to False.
        sigmoid: whether to add sigmoid function to y_pred before computation. Defaults to False.
        logit_thresh: the threshold value used to convert (after sigmoid if `sigmoid=True`)
            `y_pred` into a binary matrix. Defaults to 0.5.
        reduction: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``, ``"mean_channel"``, ``"sum_channel"``}
            Define the mode to reduce computation result of 1 batch data. Defaults to ``"mean"``.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        mutually_exclusive: bool = False,
        sigmoid: bool = False,
        logit_thresh: float = 0.5,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
    ):
        super().__init__()
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.mutually_exclusive = mutually_exclusive
        self.sigmoid = sigmoid
        self.logit_thresh = logit_thresh
        self.reduction: MetricReduction = MetricReduction(reduction)

        self.not_nans = None  # keep track for valid elements in the batch

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor):

        # compute dice (BxC) for each channel for each batch
        f = compute_meandice(
            y_pred=y_pred,
            y=y,
            include_background=self.include_background,
            to_onehot_y=self.to_onehot_y,
            mutually_exclusive=self.mutually_exclusive,
            sigmoid=self.sigmoid,
            logit_thresh=self.logit_thresh,
        )

        # some dice elements might be Nan (if ground truth y was missing (zeros))
        # we need to account for it

        nans = torch.isnan(f)
        not_nans = (~nans).float()
        f[nans] = 0

        t_zero = torch.zeros(1, device=f.device, dtype=torch.float)

        if self.reduction == MetricReduction.MEAN:
            # 2 steps, first, mean by channel (accounting for nans), then by batch

            not_nans = not_nans.sum(dim=1)
            f = torch.where(not_nans > 0, f.sum(dim=1) / not_nans, t_zero)  # channel average

            not_nans = (not_nans > 0).float().sum()
            f = torch.where(not_nans > 0, f.sum() / not_nans, t_zero)  # batch average

        elif self.reduction == MetricReduction.SUM:
            not_nans = not_nans.sum()
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == MetricReduction.MEAN_BATCH:
            not_nans = not_nans.sum(dim=0)
            f = torch.where(not_nans > 0, f.sum(dim=0) / not_nans, t_zero)  # batch average
        elif self.reduction == MetricReduction.SUM_BATCH:
            not_nans = not_nans.sum(dim=0)
            f = f.sum(dim=0)  # the batch sum
        elif self.reduction == MetricReduction.MEAN_CHANNEL:
            not_nans = not_nans.sum(dim=1)
            f = torch.where(not_nans > 0, f.sum(dim=1) / not_nans, t_zero)  # channel average
        elif self.reduction == MetricReduction.SUM_CHANNEL:
            not_nans = not_nans.sum(dim=1)
            f = f.sum(dim=1)  # the channel sum
        elif self.reduction == MetricReduction.NONE:
            pass
        else:
            raise ValueError(f"reduction={self.reduction} is invalid.")

        # save not_nans since we may need it later to know how many elements were valid
        self.not_nans = not_nans

        return f


def compute_meandice(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    include_background: bool = True,
    to_onehot_y: bool = False,
    mutually_exclusive: bool = False,
    sigmoid: bool = False,
    logit_thresh: float = 0.5,
):
    """Computes Dice score metric from full size Tensor and collects average.

    Args:
        y_pred (torch.Tensor): input data to compute, typical segmentation model output.
            it must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32].
        y (torch.Tensor): ground truth to compute mean dice metric, the first dim is batch.
            example shape: [16, 1, 32, 32] will be converted into [16, 3, 32, 32].
            alternative shape: [16, 3, 32, 32] and set `to_onehot_y=False` to use 3-class labels directly.
        include_background: whether to skip Dice computation on the first channel of
            the predicted output. Defaults to True.
        to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
        mutually_exclusive: if True, `y_pred` will be converted into a binary matrix using
            a combination of argmax and to_onehot.  Defaults to False.
        sigmoid: whether to add sigmoid function to y_pred before computation. Defaults to False.
        logit_thresh: the threshold value used to convert (after sigmoid if `sigmoid=True`)
            `y_pred` into a binary matrix. Defaults to 0.5.

    Returns:
        Dice scores per batch and per class, (shape [batch_size, n_classes]).

    Raises:
        ValueError: sigmoid=True is incompatible with mutually_exclusive=True.

    Note:
        This method provides two options to convert `y_pred` into a binary matrix
            (1) when `mutually_exclusive` is True, it uses a combination of ``argmax`` and ``to_onehot``,
            (2) when `mutually_exclusive` is False, it uses a threshold ``logit_thresh``
                (optionally with a ``sigmoid`` function before thresholding).

    """
    n_classes = y_pred.shape[1]
    n_len = len(y_pred.shape)

    if sigmoid:
        y_pred = y_pred.float().sigmoid()

    if n_classes == 1:
        if mutually_exclusive:
            warnings.warn("y_pred has only one class, mutually_exclusive=True ignored.")
        if to_onehot_y:
            warnings.warn("y_pred has only one channel, to_onehot_y=True ignored.")
        if not include_background:
            warnings.warn("y_pred has only one channel, include_background=False ignored.")
        # make both y and y_pred binary
        y_pred = (y_pred >= logit_thresh).float()
        y = (y > 0).float()
    else:  # multi-channel y_pred
        # make both y and y_pred binary
        if mutually_exclusive:
            if sigmoid:
                raise ValueError("sigmoid=True is incompatible with mutually_exclusive=True.")
            y_pred = torch.argmax(y_pred, dim=1, keepdim=True)
            y_pred = one_hot(y_pred, num_classes=n_classes)
        else:
            y_pred = (y_pred >= logit_thresh).float()
        if to_onehot_y:
            y = one_hot(y, num_classes=n_classes)

    if not include_background:
        y = y[:, 1:] if y.shape[1] > 1 else y
        y_pred = y_pred[:, 1:] if y_pred.shape[1] > 1 else y_pred

    assert y.shape == y_pred.shape, "Ground truth one-hot has differing shape (%r) from source (%r)" % (
        y.shape,
        y_pred.shape,
    )
    y = y.float()
    y_pred = y_pred.float()

    # reducing only spatial dimensions (not batch nor channels)
    reduce_axis = list(range(2, n_len))
    intersection = torch.sum(y * y_pred, dim=reduce_axis)

    y_o = torch.sum(y, reduce_axis)
    y_pred_o = torch.sum(y_pred, dim=reduce_axis)
    denominator = y_o + y_pred_o

    f = torch.where(y_o > 0, (2.0 * intersection) / denominator, torch.tensor(float("nan"), device=y_o.device))
    return f  # returns array of Dice shape: [batch, n_classes]
