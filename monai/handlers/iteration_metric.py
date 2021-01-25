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

from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Union

import torch

from monai.handlers.utils import evenly_divisible_all_gather
from monai.metrics import do_metric_reduction
from monai.utils import MetricReduction, exact_version, optional_import

NotComputableError, _ = optional_import("ignite.exceptions", "0.4.2", exact_version, "NotComputableError")
idist, _ = optional_import("ignite", "0.4.2", exact_version, "distributed")
Metric, _ = optional_import("ignite.metrics", "0.4.2", exact_version, "Metric")
reinit__is_reduced, _ = optional_import("ignite.metrics.metric", "0.4.2", exact_version, "reinit__is_reduced")
if TYPE_CHECKING:
    from ignite.engine import Engine
    from ignite.metrics import EpochWise, MetricUsage
else:
    Engine, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Engine")
    MetricUsage, _ = optional_import("ignite.metrics", "0.4.2", exact_version, "MetricUsage")
    EpochWise, _ = optional_import("ignite.metrics", "0.4.2", exact_version, "EpochWise")


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
        save_details: whether to save metric computation details per image, for example: mean_dice of every image.
            if True, will save to `engine.state.metric_details` dict with the metric name as key.

    """

    def __init__(
        self,
        metric_fn: Callable,
        output_transform: Callable = lambda x: x,
        device: Optional[torch.device] = None,
        save_details: bool = False,
    ) -> None:
        self._is_reduced: bool = False
        self.metric_fn = metric_fn
        self.save_details = save_details
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

    def compute(self) -> Any:
        """
        Raises:
            NotComputableError: When ``compute`` is called before an ``update`` occurs.

        """
        _scores = torch.cat(self._scores, dim=0)

        ws = idist.get_world_size()
        if ws > 1 and not self._is_reduced:
            # all gather across all processes
            _scores = evenly_divisible_all_gather(data=_scores, pad_dim=0)
        self._is_reduced = True

        # save score of every image into engine.state for other components
        if self.save_details:
            self.engine.state.metric_details[self.name] = _scores

        result: torch.Tensor = torch.zeros(1)
        if idist.get_rank() == 0:
            # run compute_fn on zero rank only
            result = self._reduce(_scores)

        if ws > 1:
            # broadcast result to all processes
            result = idist.broadcast(result, src=0)

        return result.item() if torch.is_tensor(result) else result

    def _reduce(self, scores) -> Any:
        return do_metric_reduction(scores, MetricReduction.MEAN)[0]

    def attach(self, engine: Engine, name: str, usage: Union[str, MetricUsage] = EpochWise()) -> None:
        """
        Attaches current metric to provided engine. On the end of engine's run,
        `engine.state.metrics` dictionary will contain computed metric's value under provided name.

        Args:
            engine: the engine to which the metric must be attached.
            name: the name of the metric to attach
            usage: the usage of the metric.

        """
        super().attach(engine=engine, name=name, usage=usage)
        # FIXME: record engine for communication, ignite will support it in the future version soon
        self.engine = engine
        self.name = name
        if self.save_details and not hasattr(engine.state, "metric_details"):
            setattr(engine.state, "metric_details", dict())
