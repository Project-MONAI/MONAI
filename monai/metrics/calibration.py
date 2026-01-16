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

from typing import Any

import torch

from monai.metrics.metric import CumulativeIterationMetric
from monai.metrics.utils import do_metric_reduction, ignore_background
from monai.utils import MetricReduction
from monai.utils.enums import StrEnum

__all__ = [
    "calibration_binning",
    "CalibrationErrorMetric",
    "CalibrationReduction",
]


def calibration_binning(
    y_pred: torch.Tensor, y: torch.Tensor, num_bins: int = 20, right: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute calibration bins for predicted probabilities and ground truth labels.
    This function calculates the mean predicted probabilities, mean ground truths,
    and bin counts for each bin using a hard binning calibration approach.

    The function operates on input and target tensors with batch and channel dimensions,
    handling each batch and channel separately. For bins that do not contain any elements,
    the mean predicted values and mean ground truth values are set to NaN.

    Args:
        y_pred: predicted tensor with shape [batch, channel, spatial], where spatial
            can be any number of dimensions. The y_pred tensor represents probabilities.
            Values should be in the range [0, 1] (probabilities).
        y: Target tensor with the same shape as y_pred. It represents ground truth values.
        num_bins: The number of bins to use for calibration. Defaults to 20. Must be >= 1.
        right: If False (default), the bins include the left boundary and exclude the right boundary.
            If True, the bins exclude the left boundary and include the right boundary.

    Returns:
        A tuple of three tensors:
            - mean_p_per_bin: Tensor of shape [batch_size, num_channels, num_bins] containing
              the mean predicted values in each bin.
            - mean_gt_per_bin: Tensor of shape [batch_size, num_channels, num_bins] containing
              the mean ground truth values in each bin.
            - bin_counts: Tensor of shape [batch_size, num_channels, num_bins] containing
              the count of elements in each bin.

    Raises:
        ValueError: If the input and target shapes do not match, if the input has fewer than 3 dimensions,
            or if num_bins < 1.

    Note:
        This function currently uses nested for loops over batch and channel dimensions
        for binning operations. Future improvements may include vectorizing these operations
        for enhanced performance.
    """
    # Input validation
    if y_pred.shape != y.shape:
        raise ValueError(f"y_pred and y must have the same shape, got {y_pred.shape} and {y.shape}.")
    if y_pred.ndim < 3:
        raise ValueError(f"y_pred must have shape (B, C, spatial...), got ndim={y_pred.ndim}.")
    if num_bins < 1:
        raise ValueError(f"num_bins must be >= 1, got {num_bins}.")

    batch_size, num_channels = y_pred.shape[:2]
    boundaries = torch.linspace(
        start=0.0,
        end=1.0 + torch.finfo(torch.float32).eps,
        steps=num_bins + 1,
        device=y_pred.device,
    )

    mean_p_per_bin = torch.zeros(batch_size, num_channels, num_bins, device=y_pred.device)
    mean_gt_per_bin = torch.zeros_like(mean_p_per_bin)
    bin_counts = torch.zeros_like(mean_p_per_bin)

    y_pred_flat = y_pred.flatten(start_dim=2).float()
    y_flat = y.flatten(start_dim=2).float()

    for b in range(batch_size):
        for c in range(num_channels):
            values_p = y_pred_flat[b, c, :]
            values_gt = y_flat[b, c, :]

            # Compute bin indices and clamp to valid range to handle out-of-range values
            bin_idx = torch.bucketize(values_p, boundaries[1:], right=right)
            bin_idx = bin_idx.clamp(max=num_bins - 1)

            # Compute bin counts using scatter_add
            counts = torch.zeros(num_bins, device=y_pred.device, dtype=torch.float32)
            counts.scatter_add_(0, bin_idx, torch.ones_like(values_p))
            bin_counts[b, c, :] = counts

            # Compute sums for mean calculation using scatter_add (more compatible than scatter_reduce)
            sum_p = torch.zeros(num_bins, device=y_pred.device, dtype=torch.float32)
            sum_p.scatter_add_(0, bin_idx, values_p)

            sum_gt = torch.zeros(num_bins, device=y_pred.device, dtype=torch.float32)
            sum_gt.scatter_add_(0, bin_idx, values_gt)

            # Compute means, avoiding division by zero
            safe_counts = counts.clamp(min=1)
            mean_p_per_bin[b, c, :] = sum_p / safe_counts
            mean_gt_per_bin[b, c, :] = sum_gt / safe_counts

    # Set empty bins to NaN
    mean_p_per_bin[bin_counts == 0] = torch.nan
    mean_gt_per_bin[bin_counts == 0] = torch.nan

    return mean_p_per_bin, mean_gt_per_bin, bin_counts


class CalibrationReduction(StrEnum):
    """
    Enumeration of calibration error reduction methods.

    - EXPECTED: Expected Calibration Error (ECE) - weighted average by bin count
    - AVERAGE: Average Calibration Error (ACE) - simple average across bins
    - MAXIMUM: Maximum Calibration Error (MCE) - maximum error across bins
    """

    EXPECTED = "expected"
    AVERAGE = "average"
    MAXIMUM = "maximum"


class CalibrationErrorMetric(CumulativeIterationMetric):
    """
    Compute the Calibration Error between predicted probabilities and ground truth labels.
    This metric is suitable for multi-class tasks and supports batched inputs.

    The input `y_pred` represents the model's predicted probabilities, and `y` represents the ground truth labels.
    `y_pred` is expected to have probabilities, and `y` should be in one-hot format. You can use suitable transforms
    in `monai.transforms.post` to achieve the desired format.

    The `include_background` parameter can be set to `False` to exclude the first category (channel index 0),
    which is conventionally assumed to be the background. This is particularly useful in segmentation tasks where
    the background class might skew the calibration results.

    The metric supports both single-channel and multi-channel data. For multi-channel data, the input tensors
    should be in the format of BCHW[D], where B is the batch size, C is the number of channels, and HW[D]
    are the spatial dimensions.

    Args:
        num_bins: Number of bins to divide probabilities into for calibration calculation. Defaults to 20.
        include_background: Whether to include computation on the first channel of the predicted output.
            Defaults to `True`.
        calibration_reduction: Method for calculating calibration error values from binned data.
            Available modes are `"expected"`, `"average"`, and `"maximum"`. Defaults to `"expected"`.
        metric_reduction: Mode of reduction to apply to the metrics.
            Reduction is only applied to non-NaN values.
            Available reduction modes are `"none"`, `"mean"`, `"sum"`, `"mean_batch"`,
            `"sum_batch"`, `"mean_channel"`, and `"sum_channel"`.
            Defaults to `"mean"`. If set to `"none"`, no reduction will be performed.
        get_not_nans: Whether to return the count of non-NaN values.
            If `True`, `aggregate()` returns a tuple (metric, not_nans). Defaults to `False`.
        right: Whether to use the right or left bin edge for binning. Defaults to `False` (left).

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Example:
        >>> from monai.transforms import Activations, AsDiscrete
        >>> # Transforms to convert model outputs to probabilities and labels to one-hot
        >>> softmax = Activations(softmax=True)  # or sigmoid=True for binary/multi-label
        >>> to_onehot = AsDiscrete(to_onehot=num_classes)
        >>> metric = CalibrationErrorMetric(num_bins=15, include_background=False, calibration_reduction="expected")
        >>> for batch_data in dataloader:
        >>>     logits, labels = model(batch_data)
        >>>     preds = softmax(logits)  # convert logits to probabilities
        >>>     labels_onehot = to_onehot(labels)  # convert labels to one-hot format
        >>>     metric(y_pred=preds, y=labels_onehot)
        >>> ece = metric.aggregate()
    """

    def __init__(
        self,
        num_bins: int = 20,
        include_background: bool = True,
        calibration_reduction: CalibrationReduction | str = CalibrationReduction.EXPECTED,
        metric_reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
        right: bool = False,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.include_background = include_background
        self.calibration_reduction = CalibrationReduction(calibration_reduction)
        self.metric_reduction = metric_reduction
        self.get_not_nans = get_not_nans
        self.right = right

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs: Any) -> torch.Tensor:  # type: ignore[override]
        """
        Compute calibration error for the given predictions and ground truth.

        Args:
            y_pred: input data to compute. It should be in the format of (batch, channel, spatial...).
                    It represents probability predictions of the model.
            y: ground truth in one-hot format. It should be in the format of (batch, channel, spatial...).
               The values should be binarized.
            **kwargs: additional keyword arguments (unused, for API compatibility).

        Returns:
            Calibration error tensor with shape (batch, channel).
        """
        if not self.include_background:
            y_pred, y = ignore_background(y_pred=y_pred, y=y)

        mean_p_per_bin, mean_gt_per_bin, bin_counts = calibration_binning(
            y_pred=y_pred, y=y, num_bins=self.num_bins, right=self.right
        )

        # Calculate the absolute differences, ignoring nan values
        abs_diff = torch.abs(mean_p_per_bin - mean_gt_per_bin)

        if self.calibration_reduction == CalibrationReduction.EXPECTED:
            # Calculate the weighted sum of absolute differences
            return torch.nansum(abs_diff * bin_counts, dim=-1) / torch.sum(bin_counts, dim=-1)
        elif self.calibration_reduction == CalibrationReduction.AVERAGE:
            return torch.nanmean(abs_diff, dim=-1)  # Average across all dimensions, ignoring nan
        elif self.calibration_reduction == CalibrationReduction.MAXIMUM:
            abs_diff_no_nan = torch.nan_to_num(abs_diff, nan=0.0)
            return torch.max(abs_diff_no_nan, dim=-1).values  # Maximum across all dimensions
        else:
            raise ValueError(f"Unsupported calibration reduction: {self.calibration_reduction}")

    def aggregate(
        self, reduction: MetricReduction | str | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Execute reduction logic for the output of `_compute_tensor`.

        Args:
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to `self.metric_reduction`. if "none", will not
                do reduction.

        Returns:
            If `get_not_nans` is True, returns a tuple (metric, not_nans), otherwise returns only the metric.
        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.metric_reduction)
        return (f, not_nans) if self.get_not_nans else f
