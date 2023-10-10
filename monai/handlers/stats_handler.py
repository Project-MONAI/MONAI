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

import logging
import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import torch

from monai.apps import get_logger
from monai.config import IgniteInfo
from monai.utils import is_scalar, min_version, optional_import

Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import(
        "ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine", as_type="decorator"
    )

DEFAULT_KEY_VAL_FORMAT = "{}: {:.4f} "
DEFAULT_TAG = "Loss"


class StatsHandler:
    """
    StatsHandler defines a set of Ignite Event-handlers for all the log printing logics.
    It can be used for any Ignite Engine(trainer, validator and evaluator).
    And it can support logging for epoch level and iteration level with pre-defined loggers.

    Note that if ``name`` is None, this class will leverage `engine.logger` as the logger, otherwise,
    ``logging.getLogger(name)`` is used. In both cases, it's important to make sure that the logging level is at least
    ``INFO``. To change the level of logging, please call ``import ignite; ignite.utils.setup_logger(name)``
    (when ``name`` is not None) or ``engine.logger = ignite.utils.setup_logger(engine.logger.name, reset=True)``
    (when ``name`` is None) before running the engine with this handler attached.

    Default behaviors:
        - When EPOCH_COMPLETED, logs ``engine.state.metrics`` using ``self.logger``.
        - When ITERATION_COMPLETED, logs
          ``self.output_transform(engine.state.output)`` using ``self.logger``.

    Usage example::

        import ignite
        import monai

        trainer = ignite.engine.Engine(lambda x, y: [0.0])  # an example trainer
        monai.handlers.StatsHandler(name="train_stats").attach(trainer)

        trainer.run(range(3), max_epochs=4)

    More details of example is available in the tutorial:
    https://github.com/Project-MONAI/tutorials/blob/master/modules/engines/unet_training_dict.py.

    """

    def __init__(
        self,
        iteration_log: bool | Callable[[Engine, int], bool] = True,
        epoch_log: bool | Callable[[Engine, int], bool] = True,
        epoch_print_logger: Callable[[Engine], Any] | None = None,
        iteration_print_logger: Callable[[Engine], Any] | None = None,
        output_transform: Callable = lambda x: x[0],
        global_epoch_transform: Callable = lambda x: x,
        state_attributes: Sequence[str] | None = None,
        name: str | None = "StatsHandler",
        tag_name: str = DEFAULT_TAG,
        key_var_format: str = DEFAULT_KEY_VAL_FORMAT,
    ) -> None:
        """

        Args:
            iteration_log: whether to log data when iteration completed, default to `True`. ``iteration_log`` can
                be also a function and it will be interpreted as an event filter
                (see https://pytorch.org/ignite/generated/ignite.engine.events.Events.html for details).
                Event filter function accepts as input engine and event value (iteration) and should return True/False.
                Event filtering can be helpful to customize iteration logging frequency.
            epoch_log: whether to log data when epoch completed, default to `True`. ``epoch_log`` can be
                also a function and it will be interpreted as an event filter. See ``iteration_log`` argument for more
                details.
            epoch_print_logger: customized callable printer for epoch level logging.
                Must accept parameter "engine", use default printer if None.
            iteration_print_logger: customized callable printer for iteration level logging.
                Must accept parameter "engine", use default printer if None.
            output_transform: a callable that is used to transform the
                ``ignite.engine.state.output`` into a scalar to print, or a dictionary of {key: scalar}.
                In the latter case, the output string will be formatted as key: value.
                By default this value logging happens when every iteration completed.
                The default behavior is to print loss from output[0] as output is a decollated list
                and we replicated loss value for every item of the decollated list.
                `engine.state` and `output_transform` inherit from the ignite concept:
                https://pytorch.org/ignite/concepts.html#state, explanation and usage example are in the tutorial:
                https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.
            global_epoch_transform: a callable that is used to customize global epoch number.
                For example, in evaluation, the evaluator engine might want to print synced epoch number
                with the trainer engine.
            state_attributes: expected attributes from `engine.state`, if provided, will extract them
                when epoch completed.
            name: identifier of `logging.logger` to use, if None, defaulting to ``engine.logger``.
            tag_name: when iteration output is a scalar, tag_name is used to print
                tag_name: scalar_value to logger. Defaults to ``'Loss'``.
            key_var_format: a formatting string to control the output string format of key: value.

        """

        self.iteration_log = iteration_log
        self.epoch_log = epoch_log
        self.epoch_print_logger = epoch_print_logger
        self.iteration_print_logger = iteration_print_logger
        self.output_transform = output_transform
        self.global_epoch_transform = global_epoch_transform
        self.state_attributes = state_attributes
        self.tag_name = tag_name
        self.key_var_format = key_var_format
        self.logger = get_logger(name)  # type: ignore
        self.name = name

    def attach(self, engine: Engine) -> None:
        """
        Register a set of Ignite Event-Handlers to a specified Ignite engine.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.

        """
        if self.name is None:
            self.logger = engine.logger
        if self.logger.getEffectiveLevel() > logging.INFO:
            suggested = f"\n\nimport ignite\nignite.utils.setup_logger('{self.logger.name}', reset=True)"
            if self.logger.name != engine.logger.name:
                suggested += f"\nignite.utils.setup_logger('{engine.logger.name}', reset=True)"
            suggested += "\n\n"
            warnings.warn(
                f"the effective log level of {self.logger.name} is higher than INFO, StatsHandler may not output logs,"
                f"\nplease use the following code before running the engine to enable it: {suggested}"
            )
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
        if not engine.has_event_handler(self.exception_raised, Events.EXCEPTION_RAISED):
            engine.add_event_handler(Events.EXCEPTION_RAISED, self.exception_raised)

    def epoch_completed(self, engine: Engine) -> None:
        """
        Handler for train or validation/evaluation epoch completed Event.
        Print epoch level log, default values are from Ignite `engine.state.metrics` dict.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.

        """
        if self.epoch_print_logger is not None:
            self.epoch_print_logger(engine)
        else:
            self._default_epoch_print(engine)

    def iteration_completed(self, engine: Engine) -> None:
        """
        Handler for train or validation/evaluation iteration completed Event.
        Print iteration level log, default values are from Ignite `engine.state.output`.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.

        """
        if self.iteration_print_logger is not None:
            self.iteration_print_logger(engine)
        else:
            self._default_iteration_print(engine)

    def exception_raised(self, _engine: Engine, e: Exception) -> None:
        """
        Handler for train or validation/evaluation exception raised Event.
        Print the exception information and traceback. This callback may be skipped because the logic
        with Ignite can only trigger the first attached handler for `EXCEPTION_RAISED` event.

        Args:
            _engine: Ignite Engine, unused argument.
            e: the exception caught in Ignite during engine.run().

        """
        self.logger.exception(f"Exception: {e}")
        raise e

    def _default_epoch_print(self, engine: Engine) -> None:
        """
        Execute epoch level log operation.
        Default to print the values from Ignite `engine.state.metrics` dict and
        print the values of specified attributes of `engine.state`.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.

        """
        current_epoch = self.global_epoch_transform(engine.state.epoch)

        prints_dict = engine.state.metrics
        if prints_dict is not None and len(prints_dict) > 0:
            out_str = f"Epoch[{current_epoch}] Metrics -- "
            for name in sorted(prints_dict):
                value = prints_dict[name]
                out_str += self.key_var_format.format(name, value) if is_scalar(value) else f"{name}: {str(value)}"
            self.logger.info(out_str)

        if (
            hasattr(engine.state, "key_metric_name")
            and hasattr(engine.state, "best_metric")
            and hasattr(engine.state, "best_metric_epoch")
            and engine.state.key_metric_name is not None
        ):
            out_str = f"Key metric: {engine.state.key_metric_name} "
            out_str += f"best value: {engine.state.best_metric} "
            out_str += f"at epoch: {engine.state.best_metric_epoch}"
            self.logger.info(out_str)

        if self.state_attributes is not None and len(self.state_attributes) > 0:
            out_str = "State values: "
            for attr in self.state_attributes:
                out_str += f"{attr}: {getattr(engine.state, attr, None)} "
            self.logger.info(out_str)

    def _default_iteration_print(self, engine: Engine) -> None:
        """
        Execute iteration log operation based on Ignite `engine.state.output` data.
        Print the values from `self.output_transform(engine.state.output)`.
        Since `engine.state.output` is a decollated list and we replicated the loss value for every item
        of the decollated list, the default behavior is to print the loss from `output[0]`.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.

        """
        loss = self.output_transform(engine.state.output)
        if loss is None:
            return  # no printing if the output is empty

        out_str = ""
        if isinstance(loss, dict):  # print dictionary items
            for name in sorted(loss):
                value = loss[name]
                if not is_scalar(value):
                    warnings.warn(
                        "ignoring non-scalar output in StatsHandler,"
                        " make sure `output_transform(engine.state.output)` returns"
                        " a scalar or dictionary of key and scalar pairs to avoid this warning."
                        " {}:{}".format(name, type(value))
                    )
                    continue  # not printing multi dimensional output
                out_str += self.key_var_format.format(name, value.item() if isinstance(value, torch.Tensor) else value)
        elif is_scalar(loss):  # not printing multi dimensional output
            out_str += self.key_var_format.format(
                self.tag_name, loss.item() if isinstance(loss, torch.Tensor) else loss
            )
        else:
            warnings.warn(
                "ignoring non-scalar output in StatsHandler,"
                " make sure `output_transform(engine.state.output)` returns"
                " a scalar or a dictionary of key and scalar pairs to avoid this warning."
                " {}".format(type(loss))
            )

        if not out_str:
            return  # no value to print

        num_iterations = engine.state.epoch_length
        current_iteration = engine.state.iteration
        if num_iterations is not None:
            current_iteration = (current_iteration - 1) % num_iterations + 1
        current_epoch = engine.state.epoch
        num_epochs = engine.state.max_epochs

        base_str = f"Epoch: {current_epoch}/{num_epochs}, Iter: {current_iteration}/{num_iterations} --"

        self.logger.info(" ".join([base_str, out_str]))
