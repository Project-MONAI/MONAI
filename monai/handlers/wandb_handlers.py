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
from typing import TYPE_CHECKING, Any, Callable, Sequence

import torch

from monai.config import IgniteInfo
from monai.utils import is_scalar, min_version, optional_import

Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")

if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import(
        "ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine", as_type="decorator"
    )

if TYPE_CHECKING:
    import wandb
else:
    wandb, _ = optional_import("wandb")


DEFAULT_TAG = "Loss"


class WandbStatsHandler:
    """
    WandbStatsHandler defines a set of Ignite Event-handlers for all the Weights & Biases logging
    logic. It can be used for any Ignite Engine(trainer, validator and evaluator) and support both
    epoch level and iteration level. The expected data source is Ignite ``engine.state.output`` and
    ``engine.state.metrics``.
    """

    def __init__(
        self,
        iteration_log: bool = True,
        epoch_log: bool = True,
        epoch_event_writer: Callable[[Engine], Any] | None = None,
        epoch_interval: int = 1,
        iteration_event_writer: Callable[[Engine], Any] | None = None,
        iteration_interval: int = 1,
        output_transform: Callable = lambda x: x[0],
        global_epoch_transform: Callable = lambda x: x,
        state_attributes: Sequence[str] | None = None,
        tag_name: str = DEFAULT_TAG,
        **kwargs,
    ):
        """
        Args:
            iteration_log: Whether to write data to Weights & Biases when iteration completed,
                default to `True`.
            epoch_log: Whether to write data to Weights & Biases when epoch completed, default to
                `True`.
            epoch_event_writer: Customized callable Weights & Biases writer for epoch level. Must
                accept the parameter "engine" and "summary_writer", use default event writer if None.
            epoch_interval: The epoch interval at which the epoch_event_writer is called. Defaults
                to 1.
            iteration_event_writer: Customized callable Weights & Biases writer for iteration level.
                Must accept parameter "engine" and "summary_writer", use default event writer if None.
            iteration_interval: The iteration interval at which the iteration_event_writer is called.
                Defaults to 1.
            output_transform: A callable that is used to transform the `ignite.engine.state.output`
                into a scalar to plot, or a dictionary of `{key: scalar}`. In the latter case, the
                output string will be formatted as key: value. By default this value plotting happens
                when every iteration completed. The default behavior is to print loss from output[0] as
                output is a decollated list and we replicated loss value for every item of the decollated
                list. `engine.state` and `output_transform` inherit from the
                ignite concept: https://pytorch.org/ignite/generated/ignite.engine.events.State.html,
                explanation and usage example are in the tutorial:
                https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.
            global_epoch_transform: A callable that is used to customize global epoch number. For example,
                in evaluation, the evaluator engine might want to use trainer engines epoch number when
                plotting epoch vs metric curves.
            state_attributes: Expected attributes from `engine.state`, if provided, will extract them when
                epoch completed.
            tag_name: When iteration output is a scalar, tag_name is used to plot, defaults to `'Loss'`.
        """
        if wandb.run is None:
            wandb.init(**kwargs)

        self.iteration_log = iteration_log
        self.epoch_log = epoch_log
        self.epoch_event_writer = epoch_event_writer
        self.epoch_interval = epoch_interval
        self.iteration_event_writer = iteration_event_writer
        self.iteration_interval = iteration_interval
        self.output_transform = output_transform
        self.global_epoch_transform = global_epoch_transform
        self.state_attributes = state_attributes
        self.tag_name = tag_name

    def attach(self, engine: Engine) -> None:
        """
        Register a set of Ignite Event-Handlers to a specified Ignite engine.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self.iteration_log and not engine.has_event_handler(self.iteration_completed, Events.ITERATION_COMPLETED):
            engine.add_event_handler(
                Events.ITERATION_COMPLETED(every=self.iteration_interval), self.iteration_completed
            )
        if self.epoch_log and not engine.has_event_handler(self.epoch_completed, Events.EPOCH_COMPLETED):
            engine.add_event_handler(Events.EPOCH_COMPLETED(every=self.epoch_interval), self.epoch_completed)

    def epoch_completed(self, engine: Engine) -> None:
        """
        Handler for train or validation/evaluation epoch completed Event. Write epoch level events
        to Weights & Biases, default values are from Ignite `engine.state.metrics` dict.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self.epoch_event_writer is not None:
            self.epoch_event_writer(engine)
        else:
            self._default_epoch_writer(engine)

    def iteration_completed(self, engine: Engine) -> None:
        """
        Handler for train or validation/evaluation iteration completed Event. Write iteration level
        events to Weighs & Biases, default values are from Ignite `engine.state.output`.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self.iteration_event_writer is not None:
            self.iteration_event_writer(engine)
        else:
            self._default_iteration_writer(engine)

    def _default_epoch_writer(self, engine: Engine) -> None:
        """
        Execute epoch level event write operation. Default to write the values from Ignite
        ``engine.state.metrics`` dict and write the values of specified attributes of ``engine.state``
        to [Weights & Biases](https://wandb.ai/site).

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        summary_dict = engine.state.metrics

        for key, value in summary_dict.items():
            if is_scalar(value):
                value = value.item() if isinstance(value, torch.Tensor) else value
                wandb.log({key: value})

        if self.state_attributes is not None:
            for attr in self.state_attributes:
                value = getattr(engine.state, attr, None)
                value = value.item() if isinstance(value, torch.Tensor) else value
                wandb.log({attr: value})

    def _default_iteration_writer(self, engine: Engine) -> None:
        """
        Execute iteration level event write operation based on Ignite ``engine.state.output`` data.
        Extract the values from ``self.output_transform(engine.state.output)``. Since
        ``engine.state.output`` is a decollated list and we replicated the loss value for every item
        of the decollated list, the default behavior is to track the loss from ``output[0]``.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        loss = self.output_transform(engine.state.output)
        if loss is None:
            return  # do nothing if output is empty
        log_dict = dict()
        if isinstance(loss, dict):
            for key, value in loss.items():
                if not is_scalar(value):
                    warnings.warn(
                        "ignoring non-scalar output in WandbStatsHandler,"
                        " make sure ``output_transform(engine.state.output)`` returns"
                        " a scalar or dictionary of key and scalar pairs to avoid this warning."
                        " {}:{}".format(key, type(value))
                    )
                    continue  # not plot multi dimensional output
                log_dict[key] = value.item() if isinstance(value, torch.Tensor) else value
        elif is_scalar(loss):  # not printing multi dimensional output
            log_dict[self.tag_name] = loss.item() if isinstance(loss, torch.Tensor) else loss
        else:
            warnings.warn(
                "ignoring non-scalar output in WandbStatsHandler,"
                " make sure ``output_transform(engine.state.output)`` returns"
                " a scalar or a dictionary of key and scalar pairs to avoid this warning."
                " {}".format(type(loss))
            )

        wandb.log(log_dict)

    def close(self):
        """
        Finish the Weights & Biases run.
        """
        if wandb.run is not None:
            wandb.finish()
