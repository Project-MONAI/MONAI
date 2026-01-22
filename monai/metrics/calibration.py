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

__all__ = ["CalibrationErrorMetric", "CalibrationReduction", "calibration_binning"]


def calibration_binning(
    y_pred: torch.Tensor, y: torch.Tensor, num_bins: int = 20, right: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute calibration bins for predicted probabilities and ground truth labels.

    This function implements hard binning for calibration analysis, grouping predictions
    into bins based on their confidence values and computing statistics for each bin.
    These statistics can be used to assess model calibration or plot reliability diagrams.

    A well-calibrated model should have predicted probabilities that match empirical accuracy.
    For example, among all predictions with 80% confidence, approximately 80% should be correct.
    This function provides the per-bin statistics needed to evaluate this property.

    The function operates on input and target tensors with batch and channel dimensions,
    handling each batch and channel separately. For bins that do not contain any elements,
    the mean predicted values and mean ground truth values are set to NaN.

    Args:
        y_pred: Predicted probabilities with shape ``(B, C, spatial...)``, where B is batch size,
            C is number of classes/channels, and spatial can be any number of dimensions (H, W, D, etc.).
            Values should be in the range [0, 1].
        y: Ground truth tensor with the same shape as ``y_pred``. Should be one-hot encoded
            or contain binary values (0 or 1) indicating the true class membership.
        num_bins: Number of equally-spaced bins to divide the [0, 1] probability range into.
            Defaults to 20. Must be >= 1.
        right: Determines bin boundary inclusion. If False (default), bins include the left
            boundary and exclude the right (i.e., [left, right)). If True, bins exclude the
            left boundary and include the right (i.e., (left, right]).

    Returns:
        A tuple of three tensors, each with shape ``(B, C, num_bins)``:
            - **mean_p_per_bin**: Mean predicted probability for samples in each bin.
            - **mean_gt_per_bin**: Mean ground truth value (empirical accuracy) for samples in each bin.
            - **bin_counts**: Number of samples falling into each bin.

        Bins with no samples have NaN values for mean_p_per_bin and mean_gt_per_bin.

    Raises:
        ValueError: If ``y_pred`` and ``y`` have different shapes, if input has fewer than
            3 dimensions, or if ``num_bins < 1``.

    References:
        - Guo, C., et al. "On Calibration of Modern Neural Networks." ICML 2017.
          https://proceedings.mlr.press/v70/guo17a.html
        - Barfoot, T., et al. "Average Calibration Losses for Reliable Uncertainty in
          Medical Image Segmentation." arXiv:2506.03942v3, 2025.
          https://arxiv.org/abs/2506.03942v3

    Note:
        This function uses nested loops over batch and channel dimensions for binning operations.
        For reliability diagram visualization, use the returned statistics to plot mean predicted
        probability vs. empirical accuracy for each bin.

    Example:
        >>> import torch
        >>> # Binary segmentation: batch=1, channels=2, spatial=4x4
        >>> y_pred = torch.rand(1, 2, 4, 4)  # predicted probabilities
        >>> y = torch.randint(0, 2, (1, 2, 4, 4)).float()  # one-hot ground truth
        >>> mean_p, mean_gt, counts = calibration_binning(y_pred, y, num_bins=10)
        >>> # mean_p, mean_gt, counts each have shape (1, 2, 10)
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
        start=0.0, end=1.0 + torch.finfo(torch.float32).eps, steps=num_bins + 1, device=y_pred.device
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
    Enumeration of calibration error reduction methods for aggregating per-bin calibration errors.

    - **EXPECTED**: Expected Calibration Error (ECE) - weighted average of per-bin errors by bin count.
      This is the most commonly used calibration metric, giving more weight to bins with more samples.
    - **AVERAGE**: Average Calibration Error (ACE) - unweighted mean of per-bin errors.
      Treats all bins equally regardless of sample count.
    - **MAXIMUM**: Maximum Calibration Error (MCE) - worst-case calibration error across all bins.
      Useful for identifying the confidence range with poorest calibration.

    References:
        - Naeini, M.P., et al. "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI 2015.
        - Guo, C., et al. "On Calibration of Modern Neural Networks." ICML 2017.
    """

    EXPECTED = "expected"
    AVERAGE = "average"
    MAXIMUM = "maximum"


class CalibrationErrorMetric(CumulativeIterationMetric):
    """
    Compute the Calibration Error between predicted probabilities and ground truth labels.

    **Why Calibration Matters:**

    A well-calibrated classifier produces probability estimates that reflect true correctness likelihood.
    For instance, if a model predicts 80% probability for class A, a well calibrated and reliable model
    should be correct approximately 80% of the time among all such predictions.
    Modern neural networks, despite high accuracy, are often poorly calibrated, as they tend to be
    overconfident in their predictions.
    This is particularly important in medical imaging where probability estimates may inform clinical decisions.

    **How It Works:**

    This metric uses a binning approach: predictions are grouped into bins based on their confidence
    (predicted probability), and for each bin, the average confidence is compared to the empirical
    accuracy (fraction of correct predictions). The calibration error measures the discrepancy between
    these values across all bins.

    Three reduction modes are supported:

    - **Expected Calibration Error (ECE)**: Weighted average of per-bin errors, where weights are
      proportional to the number of samples in each bin. Most commonly used metric.
    - **Average Calibration Error (ACE)**: Simple unweighted average across bins.
    - **Maximum Calibration Error (MCE)**: The largest calibration error among all bins.

    The metric supports both single-channel and multi-channel data in the format ``(B, C, H, W[, D])``,
    where B is batch size, C is number of classes, and H, W, D are spatial dimensions.

    Args:
        num_bins: Number of equally-spaced bins to divide the [0, 1] probability range into.
            Defaults to 20.
        include_background: Whether to include the first channel (index 0) in the computation.
            Set to ``False`` to exclude background class, which is useful in segmentation tasks
            where background may dominate and skew calibration results. Defaults to ``True``.
        calibration_reduction: Method for calculating calibration error from binned data.
            Available modes: ``"expected"`` (ECE), ``"average"`` (ACE), ``"maximum"`` (MCE).
            Defaults to ``"expected"``.
        metric_reduction: Reduction mode to apply across batch/channel dimensions after computing
            per-sample calibration errors. Available modes: ``"none"``, ``"mean"``, ``"sum"``,
            ``"mean_batch"``, ``"sum_batch"``, ``"mean_channel"``, ``"sum_channel"``.
            Defaults to ``"mean"``.
        get_not_nans: If ``True``, ``aggregate()`` returns a tuple ``(metric, not_nans)`` where
            ``not_nans`` is the count of non-NaN values. Defaults to ``False``.
        right: Bin boundary inclusion rule. If ``False`` (default), bins are ``[left, right)``.
            If ``True``, bins are ``(left, right]``.

    References:
        - Guo, C., et al. "On Calibration of Modern Neural Networks." ICML 2017.
          https://proceedings.mlr.press/v70/guo17a.html
        - Barfoot, T., et al. "Average Calibration Losses for Reliable Uncertainty in
          Medical Image Segmentation." arXiv:2506.03942v3, 2025.
          https://arxiv.org/abs/2506.03942v3

    See Also:
        - :py:class:`monai.handlers.CalibrationError`: Ignite handler wrapper for this metric.
        - :py:func:`calibration_binning`: Low-level binning function for reliability diagrams.

    Example:
        Typical execution steps follow :py:class:`monai.metrics.metric.Cumulative`.

        >>> import torch
        >>> from monai.metrics import CalibrationErrorMetric
        >>> from monai.transforms import Activations, AsDiscrete
        >>>
        >>> # Setup transforms for probability conversion
        >>> num_classes = 3
        >>> softmax = Activations(softmax=True)  # convert logits to probabilities
        >>> to_onehot = AsDiscrete(to_onehot=num_classes)  # convert labels to one-hot
        >>>
        >>> # Create metric (Expected Calibration Error, excluding background)
        >>> metric = CalibrationErrorMetric(
        ...     num_bins=15,
        ...     include_background=False,
        ...     calibration_reduction="expected"
        ... )
        >>>
        >>> # Evaluation loop
        >>> for batch_data in dataloader:
        ...     logits, labels = model(batch_data)
        ...     preds = softmax(logits)  # shape: (B, C, H, W) with values in [0, 1]
        ...     labels_onehot = to_onehot(labels)  # shape: (B, C, H, W) with values 0 or 1
        ...     metric(y_pred=preds, y=labels_onehot)
        >>>
        >>> # Get final calibration error
        >>> ece = metric.aggregate()
        >>> print(f"Expected Calibration Error: {ece:.4f}")
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
            # Handle zero denominator case (all bins empty) by returning NaN
            denom = torch.sum(bin_counts, dim=-1)
            zero_mask = denom == 0
            safe_denom = torch.where(zero_mask, torch.ones_like(denom), denom)
            result = torch.nansum(abs_diff * bin_counts, dim=-1) / safe_denom
            result = torch.where(zero_mask, torch.full_like(result, float("nan")), result)
            return result
        elif self.calibration_reduction == CalibrationReduction.AVERAGE:
            return torch.nanmean(abs_diff, dim=-1)  # Average across all dimensions, ignoring nan
        elif self.calibration_reduction == CalibrationReduction.MAXIMUM:
            # Replace NaN with -inf for max computation, then restore NaN for all-NaN cases
            abs_diff_for_max = torch.nan_to_num(abs_diff, nan=float("-inf"))
            max_vals = torch.max(abs_diff_for_max, dim=-1).values
            # Restore NaN where all bins were empty (max is -inf)
            max_vals = torch.where(max_vals == float("-inf"), torch.full_like(max_vals, float("nan")), max_vals)
            return max_vals
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
