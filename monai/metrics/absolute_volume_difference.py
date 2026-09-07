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
from monai.utils import MetricReduction

from .metric import CumulativeIterationMetric

__all__ = ["AbsoluteVolumeDifferenceMetric", "compute_absolute_volume_difference"]


class AbsoluteVolumeDifferenceMetric(CumulativeIterationMetric):
    """
    Compute the Absolute Volume Difference (AVD) between predicted and ground-truth
    segmentation masks.

    AVD measures the absolute difference in the number of foreground voxels between
    prediction and ground truth, per class.  It is particularly useful for small-object
    segmentation (e.g. retinal fluid in OCT volumes) where Dice score is known to be
    overly sensitive to volume size and does not directly reflect volume discrepancies.

    .. note::
        For 2D inputs this computes the difference in foreground **areas** rather than
        volumes.  In all cases the returned values are raw voxel/pixel counts and are
        **not** scaled by the voxel/pixel spacing, so they are not expressed in the
        physical units of the original image.

    Reference:
        Bogunovic et al. (2019). RETOUCH: The Retinal OCT Fluid Detection and
        Segmentation Benchmark and Challenge.
        IEEE Transactions on Medical Imaging, 38(8), 1858-1874.
        https://ieeexplore.ieee.org/document/8653407

    The inputs ``y_pred`` and ``y`` are expected to be binarized one-hot tensors with
    shape BCHW[D].  If they contain continuous values (e.g. sigmoid outputs), binarize
    them first with a suitable threshold transform.

    The typical execution steps of this metric class follow
    :py:class:`monai.metrics.metric.Cumulative`.

    Example:

    .. code-block:: python

        import torch
        from monai.metrics import AbsoluteVolumeDifferenceMetric

        batch_size, n_classes = 4, 3
        y_pred = torch.randint(0, 2, (batch_size, n_classes, 64, 64, 32)).float()
        y      = torch.randint(0, 2, (batch_size, n_classes, 64, 64, 32)).float()

        metric = AbsoluteVolumeDifferenceMetric(include_background=False)
        metric(y_pred, y)                 # accumulate
        result = metric.aggregate()       # shape: (n_classes - 1,) after mean reduction
        metric.reset()

    Args:
        include_background: whether to include AVD computation on the first channel
            (index 0), which is by convention assumed to be background.  Defaults to
            ``True``.  Set to ``False`` when the background class dominates and you only
            care about foreground classes (e.g. fluid sub-types in OCT).
        reduction: defines how to aggregate per-batch-per-class results.  Available
            modes are enumerated in :py:class:`monai.utils.enums.MetricReduction`.
            Defaults to ``"mean"``.
        get_not_nans: if ``True``, :meth:`aggregate` returns ``(metric, not_nans)``
            where ``not_nans`` counts the number of valid (non-NaN) values.
            Defaults to ``False``.
        ignore_empty: if ``True``, cases where the ground-truth channel is entirely
            empty (zero voxels) are excluded from aggregation by setting their value
            to ``NaN``.  If ``False``, the raw absolute difference (equal to the
            predicted volume for that class) is returned.  Defaults to ``True``.
    """

    def __init__(
        self,
        include_background: bool = True,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
        ignore_empty: bool = True,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.reduction = reduction
        self.get_not_nans = get_not_nans
        self.ignore_empty = ignore_empty

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Args:
            y_pred: binarized prediction tensor, shape BCHW[D].
            y: binarized ground-truth tensor, shape BCHW[D].

        Raises:
            ValueError: when ``y_pred`` has fewer than three dimensions.
        """
        if y_pred.ndimension() < 3:
            raise ValueError(
                f"y_pred should have at least 3 dimensions (batch, channel, spatial), got {y_pred.ndimension()}."
            )
        return compute_absolute_volume_difference(
            y_pred=y_pred, y=y, include_background=self.include_background, ignore_empty=self.ignore_empty
        )

    def aggregate(
        self, reduction: MetricReduction | str | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Execute reduction logic for the accumulated AVD values.

        Args:
            reduction: optional override for the reduction mode set at construction.
        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be a PyTorch Tensor.")

        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f


def compute_absolute_volume_difference(
    y_pred: torch.Tensor, y: torch.Tensor, include_background: bool = True, ignore_empty: bool = True
) -> torch.Tensor:
    """
    Compute the Absolute Volume Difference (AVD) for a batch of segmentation predictions.

    AVD is defined per class as::

        AVD_c = | sum_{spatial}(y_pred_c) - sum_{spatial}(y_c) |

    where the sum counts the number of foreground voxels in each channel.

    Args:
        y_pred: binarized prediction tensor with shape BCHW[D].
        y: binarized ground-truth tensor with shape BCHW[D].
        include_background: whether to include the first channel (background).
            Defaults to ``True``.
        ignore_empty: if ``True``, entries where the ground-truth channel contains no
            foreground voxels are set to ``NaN`` so they are excluded during reduction.
            Defaults to ``True``.

    Returns:
        AVD per batch item and per class, shape ``[batch_size, num_classes]``.

    Raises:
        ValueError: when ``y_pred`` and ``y`` have different shapes.
    """
    if y_pred.ndim < 3:
        raise ValueError(f"y_pred should have at least 3 dimensions (batch, channel, spatial), got {y_pred.ndim}.")

    if not include_background:
        y_pred, y = ignore_background(y_pred=y_pred, y=y)

    if y_pred.shape != y.shape:
        raise ValueError(f"y_pred and y should have the same shape, got {y_pred.shape} and {y.shape}.")

    # sum over all spatial dimensions; keep batch (dim 0) and channel (dim 1)
    reduce_axis = list(range(2, y_pred.ndim))
    vol_pred = torch.sum(y_pred, dim=reduce_axis)  # [B, C]
    vol_true = torch.sum(y, dim=reduce_axis)  # [B, C]

    avd = torch.abs(vol_pred - vol_true)  # [B, C]

    if ignore_empty:
        # mark cases with no ground-truth foreground as NaN
        avd = torch.where(vol_true > 0, avd, torch.tensor(float("nan"), device=avd.device))

    return avd
