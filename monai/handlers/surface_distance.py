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

from typing import Callable, Optional

import torch

from monai.metrics import SurfaceDistanceMetric
from monai.utils import MetricReduction
from monai.handlers.iteration_metric import IterationMetric


class SurfaceDistance(IterationMetric):
    """
    Computes surface distance from full size Tensor and collects average over batch, class-channels, iterations.
    """

    def __init__(
        self,
        include_background: bool = False,
        symmetric: bool = False,
        distance_metric: str = "euclidean",
        output_transform: Callable = lambda x: x,
        device: Optional[torch.device] = None,
    ) -> None:
        """

        Args:
            include_background: whether to include distance computation on the first channel of the predicted output.
                Defaults to ``False``.
            symmetric: whether to calculate the symmetric average surface distance between
                `seg_pred` and `seg_gt`. Defaults to ``False``.
            distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
                the metric used to compute surface distance. Defaults to ``"euclidean"``.
            output_transform: transform the ignite.engine.state.output into [y_pred, y] pair.
            device: device specification in case of distributed computation usage.

        """
        metric_fn = SurfaceDistanceMetric(
            include_background=include_background,
            symmetric=symmetric,
            distance_metric=distance_metric,
            reduction=MetricReduction.NONE,
        )
        super().__init__(metric_fn=metric_fn, output_transform=output_transform, device=device)
