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

from typing import Callable, Union

from monai.handlers.ignite_metric import IgniteMetric
from monai.metrics import PanopticQualityMetric
from monai.utils import MetricReduction


class PanopticQuality(IgniteMetric):
    """
    Computes Panoptic quality from full size Tensor and collects average over batch, class-channels, iterations.
    """

    def __init__(
        self,
        num_classes: int,
        metric_name: str = "pq",
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN_BATCH,
        match_iou: float = 0.5,
        smooth_nr: float = 1e-6,
        output_transform: Callable = lambda x: x,
        save_details: bool = True,
    ) -> None:
        """

        Args:
            num_classes: number of classes. The number should not count the background.
            metric_name: output metric. The value can be "pq", "sq" or "rq".
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to `self.reduction`. if "none", will not do reduction.
            match_iou: IOU threshould to determine the pairing between `y_pred` and `y`. Usually,
                it should >= 0.5, the pairing between instances of `y_pred` and `y` are identical.
                If set `match_iou` < 0.5, this function uses Munkres assignment to find the
                maximal amout of unique pairing.
            smooth_nr: a small constant added to the numerator to avoid zero.
            output_transform: callable to extract `y_pred` and `y` from `ignite.engine.state.output` then
                construct `(y_pred, y)` pair, where `y_pred` and `y` can be `batch-first` Tensors or
                lists of `channel-first` Tensors. the form of `(y_pred, y)` is required by the `update()`.
                `engine.state` and `output_transform` inherit from the ignite concept:
                https://pytorch.org/ignite/concepts.html#state, explanation and usage example are in the tutorial:
                https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.
            save_details: whether to save metric computation details per image, for example: panoptic quality of
                every image.
                default to True, will save to `engine.state.metric_details` dict with the metric name as key.

        See also:
            :py:meth:`monai.metrics.panoptic_quality.compute_panoptic_quality`
        """
        metric_fn = PanopticQualityMetric(
            num_classes=num_classes,
            metric_name=metric_name,
            reduction=reduction,
            match_iou=match_iou,
            smooth_nr=smooth_nr,
        )
        super().__init__(metric_fn=metric_fn, output_transform=output_transform, save_details=save_details)
