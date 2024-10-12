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

from typing import cast

import torch

from monai.metrics.utils import do_metric_reduction, ignore_background
from monai.utils import MetricReduction, convert_to_numpy, convert_to_tensor, optional_import

from .metric import CumulativeIterationMetric

BinaryPairwiseMeasures, _ = optional_import("MetricsReloaded.metrics.pairwise_measures", name="BinaryPairwiseMeasures")
MultiClassPairwiseMeasures, _ = optional_import(
    "MetricsReloaded.metrics.pairwise_measures", name="MultiClassPairwiseMeasures"
)

__all__ = ["MetricsReloadedBinary", "MetricsReloadedCategorical"]


class MetricsReloadedWrapper(CumulativeIterationMetric):
    """Base class for defining MetricsReloaded metrics as a CumulativeIterationMetric.

    Args:
        metric_name: Name of a metric from the MetricsReloaded package.
        include_background: whether to include computation on the first channel of
            the predicted output. Defaults to ``True``.
        reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
            Here `not_nans` count the number of not nans for the metric,
            thus its shape equals to the shape of the metric.

    """

    def __init__(
        self,
        metric_name: str,
        include_background: bool = True,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ) -> None:
        super().__init__()
        self.metric_name = metric_name
        self.include_background = include_background
        self.reduction = reduction
        self.get_not_nans = get_not_nans

    def aggregate(
        self, reduction: MetricReduction | str | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")
        # do metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f

    def prepare_onehot(self, y_pred, y):
        """Prepares onehot encoded input for metric call."""
        y = y.float()
        y_pred = y_pred.float()
        if not self.include_background:
            y_pred, y = ignore_background(y_pred=y_pred, y=y)
        return y_pred, y, y_pred.device


class MetricsReloadedBinary(MetricsReloadedWrapper):
    """
    Wraps the binary pairwise metrics of MetricsReloaded.

    Args:
        metric_name: Name of a binary metric from the MetricsReloaded package.
        include_background: whether to include computation on the first channel of
            the predicted output. Defaults to ``True``.
        reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
            Here `not_nans` count the number of not nans for the metric,
            thus its shape equals to the shape of the metric.

    Example:

    .. code-block:: python

        import torch
        from monai.metrics import MetricsReloadedBinary

        metric_name = "Cohens Kappa"
        metric = MetricsReloadedBinary(metric_name=metric_name)

        # first iteration
        # shape [batch=1, channel=1, 2, 2]
        y_pred = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        y = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])
        print(metric(y_pred, y))

        # second iteration
        # shape [batch=1, channel=1, 2, 2]
        y_pred = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
        y = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])
        print(metric(y_pred, y))

        # aggregate
        # shape ([batch=2, channel=1])
        print(metric.aggregate(reduction="none"))  # tensor([[0.5], [0.2]])

        # reset
        metric.reset()

    """

    def __init__(
        self,
        metric_name: str,
        include_background: bool = True,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ) -> None:
        super().__init__(
            metric_name=metric_name,
            include_background=include_background,
            reduction=reduction,
            get_not_nans=get_not_nans,
        )

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Computes a binary (single-class) MetricsReloaded metric from a batch of
        predictions and references.

        Args:
            y_pred: Prediction with dimensions (batch, channel, *spatial), where channel=1.
                The values should be binarized.
            y: Ground-truth with dimensions (batch, channel, *spatial), where channel=1.
                The values should be binarized.

        Raises:
            ValueError: when `y_pred` has less than three dimensions.
            ValueError: when second dimension ~= 1

        """
        # Preprocess
        y_pred, y, device = self.prepare_onehot(y_pred, y)

        # Sanity check
        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError(f"y_pred should have at least 3 dimensions (batch, channel, spatial), got {dims}.")
        if y_pred.shape[1] != 1 or y.shape[1] != 1:
            raise ValueError(f"y_pred.shape[1]={y_pred.shape[1]} and y.shape[1]={y.shape[1]} should be one.")

        # To numpy array
        y_pred = convert_to_numpy(y_pred)
        y = convert_to_numpy(y)

        # Create binary pairwise metric object
        bpm = BinaryPairwiseMeasures(y_pred, y, axis=tuple(range(2, dims)), smooth_dr=1e-5)

        # Is requested metric available?
        if self.metric_name not in bpm.metrics:
            raise ValueError(f"Unsupported metric: {self.metric_name}")

        # Compute metric
        metric = bpm.metrics[self.metric_name]()

        # Return metric as tensor
        return convert_to_tensor(metric, device=device)  # type: ignore[no-any-return]


class MetricsReloadedCategorical(MetricsReloadedWrapper):
    """
    Wraps the categorical pairwise metrics of MetricsReloaded.


    Args:
        metric_name: Name of a categorical metric from the MetricsReloaded package.
        include_background: whether to include computation on the first channel of
            the predicted output. Defaults to ``True``.
        reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
            Here `not_nans` count the number of not nans for the metric,
            thus its shape equals to the shape of the metric.
        smooth_dr: a small constant added to the denominator to avoid nan. OBS: should be greater than zero.

    Example:

    .. code-block:: python

        import torch
        from monai.metrics import MetricsReloadedCategorical

        metric_name = "Weighted Cohens Kappa"
        metric = MetricsReloadedCategorical(metric_name=metric_name)

        # first iteration
        # shape [bach=1, channel=3, 2, 2]
        y_pred = torch.tensor([[[[0, 0], [0, 1]], [[0, 0], [0, 0]], [[1, 1], [1, 0]]]])
        y = torch.tensor([[[[1, 0], [0, 1]], [[0, 1], [0, 0]], [[0, 0], [1, 0]]]])
        print(metric(y_pred, y))

        # second iteration
        # shape [batch=1, channel=3, 2, 2]
        y_pred = torch.tensor([[[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, 0], [0, 0]]]])
        y = torch.tensor([[[[1, 0], [0, 1]], [[0, 1], [0, 0]], [[0, 0], [1, 0]]]])
        print(metric(y_pred, y))

        # aggregate
        # shape ([batch=2, channel=1])
        print(metric.aggregate(reduction="none"))  # tensor([[0.2727], [0.6000]])

        # reset
        metric.reset()

    """

    def __init__(
        self,
        metric_name: str,
        include_background: bool = True,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
        smooth_dr: float = 1e-5,
    ) -> None:
        super().__init__(
            metric_name=metric_name,
            include_background=include_background,
            reduction=reduction,
            get_not_nans=get_not_nans,
        )
        self.smooth_dr = smooth_dr

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Computes a categorical (multi-class) MetricsReloaded metric from a batch of
        predictions and references.

        Args:
            y_pred: Prediction with dimensions (batch, channel, *spatial). The values should be
                one-hot encoded and binarized.
            y: Ground-truth with dimensions (batch, channel, *spatial). The values should be 1
                one-hot encoded and binarized.

        Raises:
            ValueError: when `y_pred` has less than three dimensions.

        """
        # Preprocess
        y_pred, y, device = self.prepare_onehot(y_pred, y)

        # Sanity check
        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError(f"y_pred should have at least 3 dimensions (batch, channel, spatial), got {dims}.")

        num_classes = y_pred.shape[1]

        # Reshape and permute for compatible dimension with MetricsReloaded
        y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1], -1)
        y_pred = y_pred.permute((0, 2, 1))
        y = y.reshape(y.shape[0], y.shape[1], -1)
        y = y.permute((0, 2, 1))
        dims = y_pred.ndimension()

        # To numpy array
        y_pred = convert_to_numpy(y_pred)
        y = convert_to_numpy(y)

        # Create categorical pairwise metric object
        bpm = MultiClassPairwiseMeasures(
            y_pred,
            y,
            axis=tuple(range(1, dims)),
            smooth_dr=self.smooth_dr,
            list_values=list(range(num_classes)),
            is_onehot=True,
        )

        # Is requested metric available?
        if self.metric_name not in bpm.metrics:
            raise ValueError(f"Unsupported metric: {self.metric_name}")

        # Compute metric
        metric = bpm.metrics[self.metric_name]()

        # Put back singleton channel dimension
        metric = metric[..., None]

        # Return metric as tensor
        return cast(torch.Tensor, convert_to_tensor(metric, device=device))
