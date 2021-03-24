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

from typing import Callable, Union

import torch

from monai.handlers.iteration_metric import IterationMetric
from monai.metrics import DiceMetric
from monai.utils import MetricReduction


class MeanDice(IterationMetric):
    """
    Computes Dice score metric from full size Tensor and collects average over batch, class-channels, iterations.
    """

    def __init__(
        self,
        include_background: bool = True,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = "cpu",
        save_details: bool = True,
    ) -> None:
        """

        Args:
            include_background: whether to include dice computation on the first channel of the predicted output.
                Defaults to True.
            output_transform: transform the ignite.engine.state.output into [y_pred, y] pair.
            device: device specification in case of distributed computation usage.
            save_details: whether to save metric computation details per image, for example: mean dice of every image.
                default to True, will save to `engine.state.metric_details` dict with the metric name as key.

        See also:
            :py:meth:`monai.metrics.meandice.compute_meandice`
        """
        metric_fn = DiceMetric(
            include_background=include_background,
            reduction=MetricReduction.NONE,
        )
        super().__init__(
            metric_fn=metric_fn,
            output_transform=output_transform,
            device=device,
            save_details=save_details,
        )
