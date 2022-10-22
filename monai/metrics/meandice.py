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

from typing import Union

import torch

from monai.config import TensorOrList
from monai.metrics.utils import do_metric_reduction, ignore_background, is_binary_tensor
from monai.networks.utils import one_hot
from monai.utils import MetricReduction

from .metric import CumulativeIterationMetric


class DiceMetric(CumulativeIterationMetric):
    """
    Compute average Dice score between two tensors. It can support both multi-classes and multi-labels tasks.
    Input `y_pred` is compared with ground truth `y`.
    `y_preds` is expected to have binarized predictions and `y` should be in one-hot format. You can use suitable transforms
    in ``monai.transforms.post`` first to achieve binarized values.
    The `include_background` parameter can be set to ``False`` to exclude
    the first category (channel index 0) which is by convention assumed to be background. If the non-background
    segmentations are small compared to the total image size they can get overwhelmed by the signal from the
    background.
    `y_preds` and `y` can be a list of channel-first Tensor (CHW[D]) or a batch-first Tensor (BCHW[D]).

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Args:
        include_background: whether to skip Dice computation on the first channel of
            the predicted output. Defaults to ``True``.
        to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
        sigmoid: if True, assumes the y_pred matches the output of DiceLoss function (with sigmoid=True),
                in which case the torch.sigmoid will be used to compute class membership followed by thresholding.
                Defaults to False.
        softmax: if True, assumes the y_pred matches the output of DiceLoss function (with softmax=True),
                in which case the torch.argmax will be used to compute class membership followed by one hot encoding.
                Defaults to False.
        reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
            Here `not_nans` count the number of not nans for the metric, thus its shape equals to the shape of the metric.
        ignore_empty: whether to ignore empty ground truth cases during calculation.
            If `True`, NaN value will be set for empty ground truth cases.
            If `False`, 1 will be set if the predictions of empty ground truth cases are also empty.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
        get_not_nans: bool = False,
        ignore_empty: bool = True,
        simple: bool = False,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.reduction = reduction
        self.get_not_nans = get_not_nans
        self.ignore_empty = ignore_empty
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.simple = simple

    def __call__(self, y_pred: TensorOrList, y: TensorOrList):
        """
        Calculates the value of Dice, if self.simple==True, simply computes the
        Dice value after reduction and returns is.

        Otherwise, follows the historical variant that accumulates metric into buffers,
        and does not return the reduced representation. One would need to call .aggregate()
        separately to return the reduced representation. Furthermore if calling this method several
        times, one need to call .reset() before .aggregate().
        .aggregate() creates syncronization between buffers (in multi gpu), which can be slow or lead to deadlock
        if each process have different logic.

        """

        if self.simple:

            if isinstance(y_pred, (list, tuple)):
                y_pred = y_pred[0]
            if isinstance(y, (list, tuple)):
                y = y[0]

            data = self._compute_tensor(y_pred=y_pred, y=y)

            f, not_nans = do_metric_reduction(data, self.reduction)
            return (f, not_nans) if self.get_not_nans else f

        return super().__call__(y_pred=y_pred, y=y)

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor):  # type: ignore
        """
        Args:
            y_pred: input data to compute, typical segmentation model output.
                It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
                should be binarized.
            y: ground truth to compute mean dice metric. It must be one-hot format and first dim is batch.
                The values should be binarized.

        Raises:
            ValueError: when `y_pred` has less than three dimensions.
        """

        if not self.softmax and not self.sigmoid:
            is_binary_tensor(y_pred, "y_pred")
            is_binary_tensor(y, "y")

        if y_pred.dim() < 3:
            raise ValueError(f"y_pred should have at least 3 dimensions (batch, channel, spatial), got {y_pred.dim()}.")

        # compute dice (BxC) for each channel for each batch
        return compute_meandice(
            y_pred=y_pred,
            y=y,
            include_background=self.include_background,
            to_onehot_y=self.to_onehot_y,
            sigmoid=self.sigmoid,
            softmax=self.softmax,
            ignore_empty=self.ignore_empty,
        )

    def aggregate(self, reduction: Union[MetricReduction, str, None] = None):
        """
        Execute reduction logic for the output of `compute_meandice`.

        Args:
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to `self.reduction`. if "none", will not do reduction.

        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f


def compute_meandice(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    include_background: bool = True,
    to_onehot_y: bool = False,
    sigmoid: bool = False,
    softmax: bool = False,
    ignore_empty: bool = True,
) -> torch.Tensor:
    """Computes Dice score metric from full size Tensor and collects average.

    Args:
        y_pred: input data to compute, typical segmentation model output.
            It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
            should be binarized.
        y: ground truth to compute mean dice metric. It must be one-hot format and first dim is batch.
            The values should be binarized.
        include_background: whether to skip Dice computation on the first channel of
            the predicted output. Defaults to True.
        to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
        sigmoid: if True, assumes the y_pred matches the output of DiceLoss function (with sigmoid=True),
                in which case the torch.sigmoid will be used to compute class membership followed by thresholding.
                Defaults to False.
        softmax: if True, assumes the y_pred matches the output of DiceLoss function (with softmax=True),
                in which case the torch.argmax will be used to compute class membership followed by one hot encoding.
                Defaults to False.
        ignore_empty: whether to ignore empty ground truth cases during calculation.
            If `True`, NaN value will be set for empty ground truth cases.
            If `False`, 1 will be set if the predictions of empty ground truth cases are also empty.

    Returns:
        Dice scores per batch and per class, (shape [batch_size, num_classes]).

    Raises:
        ValueError: when `y_pred` and `y` have different shapes.

    """

    n_pred_ch = y_pred.shape[1]

    if softmax:
        if n_pred_ch > 1:
            y_pred = torch.argmax(y_pred, dim=1, keepdim=True)
            y_pred = one_hot(y_pred, num_classes=n_pred_ch, dim=1)
    elif sigmoid:
        y_pred = (torch.sigmoid(y_pred) > 0.5).float()

    if to_onehot_y and n_pred_ch > 1 and y.shape[1] == 1:
        y = one_hot(y, num_classes=n_pred_ch, dim=1)

    if not include_background:
        y_pred, y = ignore_background(y_pred=y_pred, y=y)

    y = y.float()
    y_pred = y_pred.float()

    if y.shape != y_pred.shape:
        raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")

    # reducing only spatial dimensions (not batch nor channels)
    n_len = len(y_pred.shape)
    reduce_axis = list(range(2, n_len))
    intersection = torch.sum(y * y_pred, dim=reduce_axis)

    y_o = torch.sum(y, reduce_axis)
    y_pred_o = torch.sum(y_pred, dim=reduce_axis)
    denominator = y_o + y_pred_o

    if ignore_empty is True:
        return torch.where(y_o > 0, (2.0 * intersection) / denominator, torch.tensor(float("nan"), device=y_o.device))
    return torch.where(denominator > 0, (2.0 * intersection) / denominator, torch.tensor(1.0, device=y_o.device))


class DiceMetricSimple:
    """
    An example of Dice value calculation is simple way (without convoluted logic)
    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
        get_not_nans: bool = False,
        ignore_empty: bool = True,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.reduction = reduction
        self.get_not_nans = get_not_nans
        self.ignore_empty = ignore_empty
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor):

        data = compute_meandice(
            y_pred=y_pred,
            y=y,
            include_background=self.include_background,
            to_onehot_y=self.to_onehot_y,
            sigmoid=self.sigmoid,
            softmax=self.softmax,
            ignore_empty=self.ignore_empty,
        )

        f, not_nans = do_metric_reduction(data, self.reduction)
        return (f, not_nans) if self.get_not_nans else f
