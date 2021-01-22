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

from typing import Any, Callable, List, Optional, Sequence

import torch

from monai.metrics import do_metric_reduction
from monai.utils import MetricReduction, exact_version, optional_import

NotComputableError, _ = optional_import("ignite.exceptions", "0.4.2", exact_version, "NotComputableError")
idist, _ = optional_import("ignite", "0.4.2", exact_version, "distributed")
Metric, _ = optional_import("ignite.metrics", "0.4.2", exact_version, "Metric")
reinit__is_reduced, _ = optional_import("ignite.metrics.metric", "0.4.2", exact_version, "reinit__is_reduced")


class IterationMetric(Metric):  # type: ignore[valid-type, misc] # due to optional_import
    """
    Class for metrics that should be computed on every iteration and compute final results when epoch completed.
    Similar to the `EpochMetric` in ignite:
    https://github.com/pytorch/ignite/blob/v0.4.2/ignite/metrics/epoch_metric.py#L13.

    Args:
        metric_fn: callable function or class to compute raw metric results after every iteration.
            expect to return a Tensor with shape (batch, channel, ...) or tuple (Tensor, not_nans).
        output_transform: transform the ignite.engine.state.output into [y_pred, y] pair.
        device: device specification in case of distributed computation usage.

    """

    def __init__(
        self,
        metric_fn: Callable,
        output_transform: Callable = lambda x: x,
        device: Optional[torch.device] = None,
    ) -> None:
        self._is_reduced: bool = False
        self.metric_fn = metric_fn
        self._scores: List = []
        super().__init__(output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self._scores = []

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        """
        Args:
            output: sequence with contents [y_pred, y].

        Raises:
            ValueError: When ``output`` length is not 2. metric_fn can only support y_pred and y.

        """
        if len(output) != 2:
            raise ValueError(f"output must have length 2, got {len(output)}.")
        y_pred, y = output
        score = self.metric_fn(y_pred, y)
        if isinstance(score, (tuple, list)):
            score = score[0]
        self._scores.append(score)

    def compute(self) -> float:
        """
        Raises:
            NotComputableError: When ``compute`` is called before an ``update`` occurs.

        """
        _scores = torch.cat(self._scores, dim=0)

        ws = idist.get_world_size()

        if ws > 1 and not self._is_reduced:
            # all gather across all processes
            _scores = idist.all_gather(_scores)
        self._is_reduced = True

        result: float = 0.0
        if idist.get_rank() == 0:
            # run compute_fn on zero rank only
            result = self._reduce(_scores)

        if ws > 1:
            # broadcast result to all processes
            result = idist.broadcast(result, src=0)

        return result

    def _reduce(self, scores) -> Any:
        return do_metric_reduction(scores, MetricReduction.MEAN)[0].item()
