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
from typing import TYPE_CHECKING, Any

import torch

from monai.fl.utils.constants import ExtraItems
from monai.config import IgniteInfo
from monai.utils import is_scalar, min_version, optional_import

Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")
AnalyticsDataType, _ = optional_import("nvflare.apis.analytix", name="AnalyticsDataType")
Widget, _ = optional_import("nvflare.widgets.widget", name="Widget")

if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import(
        "ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine", as_type="decorator"
    )

DEFAULT_TAG = "Loss"


class FLStatsHandler:
    """
    FLStatsHandler defines a set of Ignite Event-handlers for all the NVFlare ``AnalyticsSender`` logics.
    It can be used for any Ignite Engine(trainer, validator and evaluator).
    And it can support both epoch level and iteration level with pre-defined AnalyticsSender event sender.
    The expected data source is Ignite ``engine.state.output`` and ``engine.state.metrics``.

    Default behaviors:
        - When EPOCH_COMPLETED, write each dictionary item in
          ``engine.state.metrics`` to TensorBoard.
        - When ITERATION_COMPLETED, write each dictionary item in
          ``self.output_transform(engine.state.output)`` to TensorBoard.

    Usage example is available in the tutorial:
    https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/unet_segmentation_3d_ignite.ipynb.

    """

    def __init__(
        self,
        stats_sender: Widget | None = None,
        iteration_log: bool | Callable[[Engine, int], bool] = True,
        epoch_log: bool | Callable[[Engine, int], bool] = True,
        output_transform: Callable = lambda x: x[0],
        global_epoch_transform: Callable = lambda x: x,
        state_attributes: Sequence[str] | None = None,
        state_attributes_type: AnalyticsDataType | None = None,
        tag_name: str = DEFAULT_TAG,
    ) -> None:
        """
        Args:
            stats_sender: user can specify AnalyticsSender.
            iteration_log: whether to send data when iteration completed, default to `True`.
                ``iteration_log`` can be also a function and it will be interpreted as an event filter
                (see https://pytorch.org/ignite/generated/ignite.engine.events.Events.html for details).
                Event filter function accepts as input engine and event value (iteration) and should return True/False.
            epoch_log: whether to send data when epoch completed, default to `True`.
                ``epoch_log`` can be also a function and it will be interpreted as an event filter.
                See ``iteration_log`` argument for more details.
            output_transform: a callable that is used to transform the
                ``ignite.engine.state.output`` into a scalar to plot, or a dictionary of {key: scalar}.
                In the latter case, the output string will be formatted as key: value.
                By default this value plotting happens when every iteration completed.
                The default behavior is to print loss from output[0] as output is a decollated list
                and we replicated loss value for every item of the decollated list.
                `engine.state` and `output_transform` inherit from the ignite concept:
                https://pytorch.org/ignite/concepts.html#state, explanation and usage example are in the tutorial:
                https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.
            global_epoch_transform: a callable that is used to customize global epoch number.
                For example, in evaluation, the evaluator engine might want to use trainer engines epoch number
                when plotting epoch vs metric curves.
            state_attributes: expected attributes from `engine.state`, if provided, will extract them
                when epoch completed.
            state_attributes_type: the type of the expected attributes from `engine.state`.
                Only required when `state_attributes` is not None.
            tag_name: when iteration output is a scalar, tag_name is used to plot, defaults to ``'Loss'``.
        """

        super().__init__()
        self._sender = stats_sender
        self.iteration_log = iteration_log
        self.epoch_log = epoch_log
        self.output_transform = output_transform
        self.global_epoch_transform = global_epoch_transform
        self.state_attributes = state_attributes
        self.state_attributes_type = state_attributes_type
        self.tag_name = tag_name

    def attach(self, engine: Engine) -> None:
        """
        Register a set of Ignite Event-Handlers to a specified Ignite engine.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.

        """
        if self.iteration_log and not engine.has_event_handler(self.iteration_completed, Events.ITERATION_COMPLETED):
            event = Events.ITERATION_COMPLETED
            if callable(self.iteration_log):  # substitute event with new one using filter callable
                event = event(event_filter=self.iteration_log)
            engine.add_event_handler(event, self.iteration_completed)
        if self.epoch_log and not engine.has_event_handler(self.epoch_completed, Events.EPOCH_COMPLETED):
            event = Events.EPOCH_COMPLETED
            if callable(self.epoch_log):  # substitute event with new one using filter callable
                event = event(event_filter=self.epoch_log)
            engine.add_event_handler(event, self.epoch_completed)

    def epoch_completed(self, engine: Engine) -> None:
        """
        Handler for train or validation/evaluation epoch completed Event.
        Write epoch level events, default values are from Ignite `engine.state.metrics` dict.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.

        """
        self._sender = engine.state.extra.get(ExtraItems.STATS_SENDER, self._sender)
        self._default_epoch_sender(engine, self._sender)

    def iteration_completed(self, engine: Engine) -> None:
        """
        Handler for train or validation/evaluation iteration completed Event.
        Write iteration level events, default values are from Ignite `engine.state.output`.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.

        """
        self._sender = engine.state.extra.get(ExtraItems.STATS_SENDER, self._sender)
        self._default_iteration_sender(engine, self._sender)

    def _send_stats(
        self, _engine: Engine, sender, tag: str, value: Any, data_type: AnalyticsDataType, step: int
    ) -> None:
        """
        Write scale value into TensorBoard.
        Default to call `Summarysender.add_scalar()`.

        Args:
            _engine: Ignite Engine, unused argument.
            sender: AnalyticsSender.
            tag: tag name in the TensorBoard.
            value: value of the scalar data for current step.
            step: index of current step.

        """
        sender._add(tag, value, data_type, step)

    def _default_epoch_sender(self, engine: Engine, sender: Widget) -> None:
        """
        Execute epoch level event write operation.
        Default to write the values from Ignite `engine.state.metrics` dict and
        write the values of specified attributes of `engine.state`.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
            sender: AnalyticsSender.

        """
        current_epoch = self.global_epoch_transform(engine.state.epoch)
        summary_dict = engine.state.metrics
        for name, value in summary_dict.items():
            self._send_stats(engine, sender, name, value, AnalyticsDataType.SCALAR, current_epoch)

        if self.state_attributes is not None:
            for attr in self.state_attributes:
                self._send_stats(engine, sender, attr, getattr(engine.state, attr, None), self.state_attributes_type, current_epoch)
        sender.flush()

    def _default_iteration_sender(self, engine: Engine, sender: Widget) -> None:
        """
        Execute iteration level event write operation based on Ignite `engine.state.output` data.
        Extract the values from `self.output_transform(engine.state.output)`.
        Since `engine.state.output` is a decollated list and we replicated the loss value for every item
        of the decollated list, the default behavior is to track the loss from `output[0]`.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
            sender: AnalyticsSender.

        """
        loss = self.output_transform(engine.state.output)
        if loss is None:
            return  # do nothing if output is empty
        if isinstance(loss, dict):
            data_type = AnalyticsDataType.SCALARS
        elif is_scalar(loss):  # not printing multi dimensional output
            data_type = AnalyticsDataType.SCALAR
        else:
            warnings.warn(
                "ignoring non-scalar output in FLStatsHandler,"
                " make sure `output_transform(engine.state.output)` returns"
                " a scalar or a dictionary of key and scalar pairs to avoid this warning."
                " {}".format(type(loss))
            )

        self._send_stats(
            _engine=engine,
            sender=sender,
            tag=self.tag_name,
            value=loss.item() if isinstance(loss, torch.Tensor) else loss,
            data_type=data_type,
            step=engine.state.iteration,
        )
        sender.flush()
