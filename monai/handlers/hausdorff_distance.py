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

from typing import Callable, Optional, Union

from monai.handlers.ignite_metric import IgniteMetric
from monai.metrics import HausdorffDistanceMetric
from monai.utils import MetricReduction


class HausdorffDistance(IgniteMetric):
    """
    Computes Hausdorff distance from full size Tensor and collects average over batch, class-channels, iterations.
    """

    def __init__(
        self,
        include_background: bool = False,
        distance_metric: str = "euclidean",
        percentile: Optional[float] = None,
        directed: bool = False,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
        output_transform: Callable = lambda x: x,
        save_details: bool = True,
    ) -> None:
        """

        Args:
            include_background: whether to include distance computation on the first channel of the predicted output.
                Defaults to ``False``.
            distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
                the metric used to compute surface distance. Defaults to ``"euclidean"``.
            percentile: an optional float number between 0 and 100. If specified, the corresponding
                percentile of the Hausdorff Distance rather than the maximum result will be achieved.
                Defaults to ``None``.
            directed: whether to calculate directed Hausdorff distance. Defaults to ``False``.
            reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
            output_transform: callable to extract `y_pred` and `y` from `ignite.engine.state.output` then
                construct `(y_pred, y)` pair, where `y_pred` and `y` can be `batch-first` Tensors or
                lists of `channel-first` Tensors. the form of `(y_pred, y)` is required by the `update()`.
                `engine.state` and `output_transform` inherit from the ignite concept:
                https://pytorch.org/ignite/concepts.html#state, explanation and usage example are in the tutorial:
                https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.
            save_details: whether to save metric computation details per image, for example: hausdorff distance
                of every image. default to True, will save to `engine.state.metric_details` dict with the metric name as key.

        """
        metric_fn = HausdorffDistanceMetric(
            include_background=include_background,
            distance_metric=distance_metric,
            percentile=percentile,
            directed=directed,
            reduction=reduction,
        )
        super().__init__(metric_fn=metric_fn, output_transform=output_transform, save_details=save_details)
