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

from monai.metrics.utils import do_metric_reduction, ignore_background
from monai.utils import MetricReduction, Weight, look_up_option

from .metric import CumulativeIterationMetric
from .utils import is_binary_tensor

__all__ = ["GeneralizedDiceScore", "compute_generalized_dice"]


class GeneralizedDiceScore(CumulativeIterationMetric):
    """Compute the Generalized Dice Score metric between tensors, as the complement of the Generalized Dice Loss defined in:

    Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017.

    The inputs `y_pred` and `y` are expected to be one-hot, binarized channel-first
    or batch-first tensors, i.e., CHW[D] or BCHW[D].

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Args:
        include_background (bool, optional): whether to include the background class (assumed to be in channel 0), in the
            score computation. Defaults to True.
        reduction (str, optional): define mode of reduction to the metrics. Available reduction modes:
            {``"none"``, ``"mean_batch"``, ``"sum_batch"``}. Default to ``"mean_batch"``. If "none", will not do reduction.
        weight_type (Union[Weight, str], optional): {``"square"``, ``"simple"``, ``"uniform"``}. Type of function to transform
            ground truth volume into a weight factor. Defaults to ``"square"``.

    Raises:
        ValueError: when the `weight_type` is not one of {``"none"``, ``"mean"``, ``"sum"``}.
    """

    def __init__(
        self,
        include_background: bool = True,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN_BATCH,
        weight_type: Union[Weight, str] = Weight.SQUARE,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        reduction_options = [
            "none",
            "mean_batch",
            "sum_batch",
            MetricReduction.NONE,
            MetricReduction.MEAN_BATCH,
            MetricReduction.SUM_BATCH,
        ]
        self.reduction = reduction
        if self.reduction not in reduction_options:
            raise ValueError(f"reduction must be one of {reduction_options}")
        self.weight_type = look_up_option(weight_type, Weight)

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor):  # type: ignore
        """Computes the Generalized Dice Score and returns a tensor with its per image values.

        Args:
            y_pred (torch.Tensor): binarized segmentation model output. It must be in one-hot format and in the NCHW[D] format,
                where N is the batch dimension, C is the channel dimension, and the remaining are the spatial dimensions.
            y (torch.Tensor): binarized ground-truth. It must be in one-hot format and have the same shape as `y_pred`.

        Raises:
            ValueError: if `y_pred` or `y` is not a binarized PyTorch tensor, if `y_pred` and `y` have less than
            three dimensions, or `y_pred` and `y` don't have the same shape.
        """
        return compute_generalized_dice(
            y_pred=y_pred, y=y, include_background=self.include_background, weight_type=self.weight_type
        )

    def aggregate(self, reduction: Union[MetricReduction, str, None] = None):
        """
        Execute reduction logic for the output of `compute_generalized_dice`.

        Args:
            reduction (Union[MetricReduction, str, None], optional): define mode of reduction to the metrics.
                Available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``}.
                Defaults to ``"mean"``. If "none", will not do reduction.
        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("The data to aggregate must be a PyTorch Tensor.")

        # Validate reduction argument if specified
        if reduction is not None:
            reduction_options = ["none", "mean", "sum", "mean_batch", "sum_batch"]
            if reduction not in reduction_options:
                raise ValueError(f"reduction must be one of {reduction_options}")

        # Do metric reduction and return
        f, _ = do_metric_reduction(data, reduction or self.reduction)

        return f


def compute_generalized_dice(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    include_background: bool = True,
    weight_type: Union[Weight, str] = Weight.SQUARE,
) -> torch.Tensor:
    """Computes the Generalized Dice Score and returns a tensor with its per image values.

    Args:
        y_pred (torch.Tensor): binarized segmentation model output. It should be binarized, in one-hot format
            and in the NCHW[D] format, where N is the batch dimension, C is the channel dimension, and the
            remaining are the spatial dimensions.
        y (torch.Tensor): binarized ground-truth. It should be binarized, in one-hot format and have the same shape as `y_pred`.
        include_background (bool, optional): whether to skip score computation on the first channel of the
            predicted output. Defaults to True.
        weight_type (Union[Weight, str], optional): {``"square"``, ``"simple"``, ``"uniform"``}. Type of function to
            transform ground truth volume into a weight factor. Defaults to ``"square"``.

    Returns:
        torch.Tensor: per batch and per class Generalized Dice Score, i.e., with the shape [batch_size, num_classes].

    Raises:
        ValueError: if `y_pred` or `y` are not PyTorch tensors, if `y_pred` and `y` have less than three dimensions,
            or `y_pred` and `y` don't have the same shape.
    """
    # Ensure tensors are binarized
    is_binary_tensor(y_pred, "y_pred")
    is_binary_tensor(y, "y")

    # Ensure tensors have at least 3 dimensions and have the same shape
    dims = y_pred.dim()
    if dims < 3:
        raise ValueError(f"y_pred should have at least 3 dimensions (batch, channel, spatial), got {dims}.")
    if y.shape != y_pred.shape:
        raise ValueError(f"y_pred - {y_pred.shape} - and y - {y.shape} - should have the same shapes.")

    # Ignore background, if needed
    if not include_background:
        y_pred, y = ignore_background(y_pred=y_pred, y=y)

    # Reducing only spatial dimensions (not batch nor channels), compute the intersection and non-weighted denominator
    reduce_axis = list(range(2, y_pred.dim()))
    intersection = torch.sum(y * y_pred, dim=reduce_axis)
    y_o = torch.sum(y, dim=reduce_axis)
    y_pred_o = torch.sum(y_pred, dim=reduce_axis)
    denominator = y_o + y_pred_o

    # Set the class weights
    weight_type = look_up_option(weight_type, Weight)
    if weight_type == Weight.SIMPLE:
        w = torch.reciprocal(y_o.float())
    elif weight_type == Weight.SQUARE:
        w = torch.reciprocal(y_o.float() * y_o.float())
    else:
        w = torch.ones_like(y_o.float())

    # Replace infinite values for non-appearing classes by the maximum weight
    for b in w:
        infs = torch.isinf(b)
        b[infs] = 0
        b[infs] = torch.max(b)

    # Compute the weighted numerator and denominator, summing along the class axis
    numer = 2.0 * (intersection * w).sum(dim=1)
    denom = (denominator * w).sum(dim=1)

    # Compute the score
    generalized_dice_score = numer / denom

    # Handle zero deivision. Where denom == 0 and the prediction volume is 0, score is 1.
    # Where denom == 0 but the prediction volume is not 0, score is 0
    y_pred_o = y_pred_o.sum(dim=-1)
    denom_zeros = denom == 0
    generalized_dice_score[denom_zeros] = torch.where(
        (y_pred_o == 0)[denom_zeros], torch.tensor(1.0), torch.tensor(0.0)
    )

    return generalized_dice_score
