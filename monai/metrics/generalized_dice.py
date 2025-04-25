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

import torch

from monai.metrics.utils import do_metric_reduction, ignore_background
from monai.utils import MetricReduction, Weight, deprecated_arg, look_up_option

from .metric import CumulativeIterationMetric


class GeneralizedDiceScore(CumulativeIterationMetric):
    """
    Compute the Generalized Dice Score metric between tensors.

    This metric is the complement of the Generalized Dice Loss defined in:
    Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
    loss function for highly unbalanced segmentations. DLMIA 2017.

    The inputs `y_pred` and `y` are expected to be one-hot, binarized batch-first tensors, i.e., NCHW[D].

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Args:
        include_background: Whether to include the background class (assumed to be in channel 0) in the
            score computation. Defaults to True.
        reduction: Define mode of reduction to the metrics. Available reduction modes:
            {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
            Default value is changed from `MetricReduction.MEAN_BATCH` to `MetricReduction.MEAN` in v1.5.0.
            Old versions computed `mean` when `mean_batch` was provided due to bug in reduction.
        weight_type: {``"square"``, ``"simple"``, ``"uniform"``}. Type of function to transform
            ground truth volume into a weight factor. Defaults to ``"square"``.

    Raises:
        ValueError: When the `reduction` is not one of MetricReduction enum.
    """

    def __init__(
        self,
        include_background: bool = True,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        weight_type: Weight | str = Weight.SQUARE,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.reduction = look_up_option(reduction, MetricReduction)
        self.weight_type = look_up_option(weight_type, Weight)
        self.sum_over_classes = self.reduction in {
            MetricReduction.SUM,
            MetricReduction.MEAN,
            MetricReduction.MEAN_CHANNEL,
            MetricReduction.SUM_CHANNEL,
        }

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Computes the Generalized Dice Score and returns a tensor with its per image values.

        Args:
            y_pred (torch.Tensor): Binarized segmentation model output. It must be in one-hot format and in the NCHW[D] format,
                where N is the batch dimension, C is the channel dimension, and the remaining are the spatial dimensions.
            y (torch.Tensor): Binarized ground-truth. It must be in one-hot format and have the same shape as `y_pred`.

        Returns:
            torch.Tensor: Generalized Dice Score averaged across batch and class

        Raises:
            ValueError: If `y_pred` and `y` have less than 3 dimensions, or `y_pred` and `y` don't have the same shape.
        """
        return compute_generalized_dice(
            y_pred=y_pred,
            y=y,
            include_background=self.include_background,
            weight_type=self.weight_type,
            sum_over_classes=self.sum_over_classes,
        )

    @deprecated_arg(
        "reduction",
        since="1.3.3",
        removed="1.7.0",
        msg_suffix="Reduction will be ignored. Set reduction during init. as gen.dice needs it during compute",
    )
    def aggregate(self, reduction: MetricReduction | str | None = None) -> torch.Tensor:
        """
        Execute reduction logic for the output of `compute_generalized_dice`.

        Returns:
            torch.Tensor: Aggregated metric value.

        Raises:
            ValueError: If the data to aggregate is not a PyTorch Tensor.
        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("The data to aggregate must be a PyTorch Tensor.")

        # Do metric reduction and return
        f, _ = do_metric_reduction(data, self.reduction)

        return f


def compute_generalized_dice(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    include_background: bool = True,
    weight_type: Weight | str = Weight.SQUARE,
    sum_over_classes: bool = False,
) -> torch.Tensor:
    """
    Computes the Generalized Dice Score and returns a tensor with its per image values.

    Args:
        y_pred (torch.Tensor): Binarized segmentation model output. It should be binarized, in one-hot format
            and in the NCHW[D] format, where N is the batch dimension, C is the channel dimension, and the
            remaining are the spatial dimensions.
        y (torch.Tensor): Binarized ground-truth. It should be binarized, in one-hot format and have the same shape as `y_pred`.
        include_background: Whether to include score computation on the first channel of the
            predicted output. Defaults to True.
        weight_type (Union[Weight, str], optional): {``"square"``, ``"simple"``, ``"uniform"``}. Type of function to
            transform ground truth volume into a weight factor. Defaults to ``"square"``.
        sum_over_labels (bool): Whether to sum the numerator and denominator across all labels before the final computation.

    Returns:
        torch.Tensor: Per batch and per class Generalized Dice Score, i.e., with the shape [batch_size, num_classes].

    Raises:
        ValueError: If `y_pred` or `y` are not PyTorch tensors, if `y_pred` and `y` have less than three dimensions,
            or `y_pred` and `y` don't have the same shape.
    """
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

    # Compute the weighted numerator and denominator, summing along the class axis when sum_over_classes is True
    if sum_over_classes:
        numer = 2.0 * (intersection * w).sum(dim=1, keepdim=True)
        denom = (denominator * w).sum(dim=1, keepdim=True)
        y_pred_o = y_pred_o.sum(dim=-1, keepdim=True)
    else:
        numer = 2.0 * (intersection * w)
        denom = denominator * w
        y_pred_o = y_pred_o

    # Compute the score
    generalized_dice_score = numer / denom

    # Handle zero division. Where denom == 0 and the prediction volume is 0, score is 1.
    # Where denom == 0 but the prediction volume is not 0, score is 0
    denom_zeros = denom == 0
    generalized_dice_score[denom_zeros] = torch.where(
        (y_pred_o == 0)[denom_zeros],
        torch.tensor(1.0, device=generalized_dice_score.device),
        torch.tensor(0.0, device=generalized_dice_score.device),
    )

    return generalized_dice_score
