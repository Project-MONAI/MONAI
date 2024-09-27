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

import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

import torch
from torch.nn.modules.loss import _Loss

from monai.metrics import CumulativeIterationMetric, LossMetric
from monai.utils import IgniteInfo, MetricReduction, min_version, optional_import

idist, _ = optional_import("ignite", IgniteInfo.OPT_IMPORT_VERSION, min_version, "distributed")

if TYPE_CHECKING:
    try:
        _, has_ignite = optional_import("ignite")
        from ignite.engine import Engine
        from ignite.metrics import Metric
        from ignite.metrics.metric import reinit__is_reduced
    except ImportError:
        has_ignite = False

else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")
    Metric, _ = optional_import("ignite.metrics", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Metric", as_type="base")
    reinit__is_reduced, _ = optional_import(
        "ignite.metrics.metric", IgniteInfo.OPT_IMPORT_VERSION, min_version, "reinit__is_reduced", as_type="decorator"
    )


class IgniteMetricHandler(Metric):
    """
    Base Metric class based on ignite event handler mechanism.
    The input `prediction` or `label` data can be a PyTorch Tensor or numpy array with batch dim and channel dim,
    or a list of PyTorch Tensor or numpy array without batch dim.

    Args:
        metric_fn: callable function or class to compute raw metric results after every iteration.
            expect to return a Tensor with shape (batch, channel, ...) or tuple (Tensor, not_nans).
        loss_fn: A torch _Loss function which is used to generate the LossMetric
        output_transform: callable to extract `y_pred` and `y` from `ignite.engine.state.output` then
            construct `(y_pred, y)` pair, where `y_pred` and `y` can be `batch-first` Tensors or
            lists of `channel-first` Tensors. the form of `(y_pred, y)` is required by the `update()`.
            `engine.state` and `output_transform` inherit from the ignite concept:
            https://pytorch.org/ignite/concepts.html#state, explanation and usage example are in the tutorial:
            https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.
        save_details: whether to save metric computation details per image, for example: mean_dice of every image.
            default to True, will save to `engine.state.metric_details` dict with the metric name as key.
        reduction: Argument for the LossMetric, look there for details
        get_not_nans: Argument for the LossMetric, look there for details

    """

    def __init__(
        self,
        metric_fn: CumulativeIterationMetric | None = None,
        loss_fn: _Loss | None = None,
        output_transform: Callable = lambda x: x,
        save_details: bool = True,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ) -> None:
        self._is_reduced: bool = False
        self.metric_fn: CumulativeIterationMetric = cast(CumulativeIterationMetric, metric_fn)
        self.loss_fn = loss_fn
        self.save_details = save_details
        self._scores: list = []
        self._engine: Engine | None = None
        self._name: str | None = None

        if self.metric_fn is None and self.loss_fn is None:
            raise ValueError("Either metric_fn or loss_fn have to be passed.")
        if self.metric_fn is not None and self.loss_fn is not None:
            raise ValueError("Either metric_fn or loss_fn have to be passed, but not both.")
        if self.loss_fn:
            self.metric_fn = LossMetric(loss_fn=self.loss_fn, reduction=reduction, get_not_nans=get_not_nans)

        super().__init__(output_transform)

    @reinit__is_reduced
    def reset(self) -> None:
        self.metric_fn.reset()

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

        self.metric_fn(y_pred, y)

    def compute(self) -> Any:
        """
        Raises:
            NotComputableError: When ``compute`` is called before an ``update`` occurs.

        """
        result = self.metric_fn.aggregate()
        if isinstance(result, (tuple, list)):
            if len(result) > 1:
                warnings.warn("metric handler can only record the first value of result list.")
            result = result[0]

        self._is_reduced = True

        # save score of every image into engine.state for other components
        if self.save_details:
            if self._engine is None or self._name is None:
                raise RuntimeError("please call the attach() function to connect expected engine first.")
            self._engine.state.metric_details[self._name] = self.metric_fn.get_buffer()  # type: ignore

        if isinstance(result, torch.Tensor):
            result = result.squeeze()
            if result.ndim == 0:
                result = result.item()
        return result

    def attach(self, engine: Engine, name: str) -> None:  # type: ignore[override]
        """
        Attaches current metric to provided engine. On the end of engine's run,
        `engine.state.metrics` dictionary will contain computed metric's value under provided name.

        Args:
            engine: the engine to which the metric must be attached.
            name: the name of the metric to attach.

        """
        super().attach(engine=engine, name=name)
        # FIXME: record engine for communication, ignite will support it in the future version soon
        self._engine = engine
        self._name = name
        if self.save_details and not hasattr(engine.state, "metric_details"):
            engine.state.metric_details = {}  # type: ignore
