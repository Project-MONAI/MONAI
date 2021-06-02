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
from monai.metrics import ConfusionMatrixMetric
from monai.metrics.utils import MetricReduction


class ConfusionMatrix(IterationMetric):
    """
    Compute confusion matrix related metrics from full size Tensor and collects average over batch, class-channels, iterations.
    """

    def __init__(
        self,
        include_background: bool = True,
        metric_name: str = "hit_rate",
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = "cpu",
        save_details: bool = True,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
    ) -> None:
        """

        Args:
            include_background: whether to skip metric computation on the first channel of
                the predicted output. Defaults to True.
            metric_name: [``"sensitivity"``, ``"specificity"``, ``"precision"``, ``"negative predictive value"``,
                ``"miss rate"``, ``"fall out"``, ``"false discovery rate"``, ``"false omission rate"``,
                ``"prevalence threshold"``, ``"threat score"``, ``"accuracy"``, ``"balanced accuracy"``,
                ``"f1 score"``, ``"matthews correlation coefficient"``, ``"fowlkes mallows index"``,
                ``"informedness"``, ``"markedness"``]
                Some of the metrics have multiple aliases (as shown in the wikipedia page aforementioned),
                and you can also input those names instead.
            output_transform: transform the ignite.engine.state.output into [y_pred, y] pair.
            device: device specification in case of distributed computation usage.
            save_details: whether to save metric computation details per image, for example: TP/TN/FP/FN of every image.
                default to True, will save to `engine.state.metric_details` dict with the metric name as key.
            reduction: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}
                Define the mode to reduce computation result. Defaults to ``"mean"``.

        See also:
            :py:meth:`monai.metrics.confusion_matrix`
        """
        metric_fn = ConfusionMatrixMetric(
            include_background=include_background,
            metric_name=metric_name,
            compute_sample=False,
            reduction=reduction,
        )
        self.metric_name = metric_name
        super().__init__(
            metric_fn=metric_fn,
            output_transform=output_transform,
            device=device,
            save_details=save_details,
        )
