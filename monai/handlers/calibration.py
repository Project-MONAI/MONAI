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
from monai.metrics import CalibrationErrorMetric, CalibrationReduction
from monai.utils import MetricReduction

__all__ = ["CalibrationError"]


class CalibrationError(IgniteMetricHandler):
    """
    Computes Calibration Error and reports the aggregated value according to `metric_reduction`
    over all accumulated iterations. Can return the expected, average, or maximum calibration error.

    Args:
        num_bins: number of bins to calculate calibration. Defaults to 20.
        include_background: whether to include calibration error computation on the first channel of
            the predicted output. Defaults to True.
        calibration_reduction: Method for calculating calibration error values from binned data.
            Available modes are `"expected"`, `"average"`, and `"maximum"`. Defaults to `"expected"`.
        metric_reduction: Mode of reduction to apply to the metrics.
            Reduction is only applied to non-NaN values.
            Available reduction modes are `"none"`, `"mean"`, `"sum"`, `"mean_batch"`,
            `"sum_batch"`, `"mean_channel"`, and `"sum_channel"`.
            Defaults to `"mean"`. If set to `"none"`, no reduction will be performed.
        output_transform: callable to extract `y_pred` and `y` from `ignite.engine.state.output` then
            construct `(y_pred, y)` pair, where `y_pred` and `y` can be `batch-first` Tensors or
            lists of `channel-first` Tensors. the form of `(y_pred, y)` is required by the `update()`.
            `engine.state` and `output_transform` inherit from the ignite concept:
            https://pytorch.org/ignite/concepts.html#state, explanation and usage example are in the tutorial:
            https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.
        save_details: whether to save metric computation details per image, for example: calibration error
            of every image. default to True, will save to `engine.state.metric_details` dict with the
            metric name as key.

    """

    def __init__(
        self,
        num_bins: int = 20,
        include_background: bool = True,
        calibration_reduction: CalibrationReduction | str = CalibrationReduction.EXPECTED,
        metric_reduction: MetricReduction | str = MetricReduction.MEAN,
        output_transform: Callable = lambda x: x,
        save_details: bool = True,
    ) -> None:
        metric_fn = CalibrationErrorMetric(
            num_bins=num_bins,
            include_background=include_background,
            calibration_reduction=calibration_reduction,
            metric_reduction=metric_reduction,
        )

        super().__init__(
            metric_fn=metric_fn,
            output_transform=output_transform,
            save_details=save_details,
        )
