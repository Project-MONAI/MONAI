# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable

from monai.handlers.ignite_metric import IgniteMetric
from monai.metrics import SurfaceDistanceMetric
from monai.utils import MetricReduction


class SurfaceDistance(IgniteMetric):
    """
    Computes surface distance from full size Tensor and collects average over batch, class-channels, iterations.
    """

    def __init__(
        self,
        include_background: bool = False,
        symmetric: bool = False,
        distance_metric: str = "euclidean",
        output_transform: Callable = lambda x: x,
        save_details: bool = True,
    ) -> None:
        """

        Args:
            include_background: whether to include distance computation on the first channel of the predicted output.
                Defaults to ``False``.
            symmetric: whether to calculate the symmetric average surface distance between
                `seg_pred` and `seg_gt`. Defaults to ``False``.
            distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
                the metric used to compute surface distance. Defaults to ``"euclidean"``.
            output_transform: callable to extract `y_pred` and `y` from `ignite.engine.state.output` then
                construct `(y_pred, y)` pair, where `y_pred` and `y` can be `batch-first` Tensors or
                lists of `channel-first` Tensors. the form of `(y_pred, y)` is required by the `update()`.
                for example: if `ignite.engine.state.output` is `{"pred": xxx, "label": xxx, "other": xxx}`,
                output_transform can be `lambda x: (x["pred"], x["label"])`.
            save_details: whether to save metric computation details per image, for example: surface dice
                of every image. default to True, will save to `engine.state.metric_details` dict with the metric name as key.

        """
        metric_fn = SurfaceDistanceMetric(
            include_background=include_background,
            symmetric=symmetric,
            distance_metric=distance_metric,
            reduction=MetricReduction.MEAN,
        )
        super().__init__(
            metric_fn=metric_fn,
            output_transform=output_transform,
            save_details=save_details,
        )
