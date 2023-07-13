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

from collections.abc import Callable

from monai.handlers.ignite_metric import IgniteMetricHandler
from monai.metrics import MetricsReloadedBinary, MetricsReloadedCategorical
from monai.utils.enums import MetricReduction


class MetricsReloadedBinaryHandler(IgniteMetricHandler):
    """
    Handler of MetricsReloadedBinary, which wraps the binary pairwise metrics of MetricsReloaded.
    """

    def __init__(
        self,
        metric_name: str,
        include_background: bool = True,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
        output_transform: Callable = lambda x: x,
        save_details: bool = True,
    ) -> None:
        """

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
            output_transform: callable to extract `y_pred` and `y` from `ignite.engine.state.output` then
                construct `(y_pred, y)` pair, where `y_pred` and `y` can be `batch-first` Tensors or
                lists of `channel-first` Tensors. the form of `(y_pred, y)` is required by the `update()`.
                `engine.state` and `output_transform` inherit from the ignite concept:
                https://pytorch.org/ignite/concepts.html#state, explanation and usage example are in the tutorial:
                https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.
            save_details: whether to save metric computation details per image, for example: TP/TN/FP/FN of every image.
                default to True, will save to `engine.state.metric_details` dict with the metric name as key.

        See also:
            :py:meth:`monai.metrics.wrapper`
        """
        metric_fn = MetricsReloadedBinary(
            metric_name=metric_name,
            include_background=include_background,
            reduction=reduction,
            get_not_nans=get_not_nans,
        )
        super().__init__(metric_fn=metric_fn, output_transform=output_transform, save_details=save_details)


class MetricsReloadedCategoricalHandler(IgniteMetricHandler):
    """
    Handler of MetricsReloadedCategorical, which wraps the categorical pairwise metrics of MetricsReloaded.
    """

    def __init__(
        self,
        metric_name: str,
        include_background: bool = True,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
        smooth_dr: float = 1e-5,
        output_transform: Callable = lambda x: x,
        save_details: bool = True,
    ) -> None:
        """

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
            output_transform: callable to extract `y_pred` and `y` from `ignite.engine.state.output` then
                construct `(y_pred, y)` pair, where `y_pred` and `y` can be `batch-first` Tensors or
                lists of `channel-first` Tensors. the form of `(y_pred, y)` is required by the `update()`.
                `engine.state` and `output_transform` inherit from the ignite concept:
                https://pytorch.org/ignite/concepts.html#state, explanation and usage example are in the tutorial:
                https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.
            save_details: whether to save metric computation details per image, for example: TP/TN/FP/FN of every image.
                default to True, will save to `engine.state.metric_details` dict with the metric name as key.

        See also:
            :py:meth:`monai.metrics.wrapper`
        """
        metric_fn = MetricsReloadedCategorical(
            metric_name=metric_name,
            include_background=include_background,
            reduction=reduction,
            get_not_nans=get_not_nans,
            smooth_dr=smooth_dr,
        )
        super().__init__(metric_fn=metric_fn, output_transform=output_transform, save_details=save_details)
