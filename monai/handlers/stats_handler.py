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

import logging
import warnings
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import torch

from monai.config import IgniteInfo
from monai.utils import is_scalar, min_version, optional_import

Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine", as_type="decorator")

DEFAULT_KEY_VAL_FORMAT = "{}: {:.4f} "
DEFAULT_TAG = "Loss"


class StatsHandler:
    """
    StatsHandler defines a set of Ignite Event-handlers for all the log printing logics.
    It can be used for any Ignite Engine(trainer, validator and evaluator).
    And it can support logging for epoch level and iteration level with pre-defined loggers.

    Note that if `name` arg is None, will leverage `engine.logger` as default logger directly, otherwise,
    get logger from `logging.getLogger(name)`, we can setup a logger outside first with the same `name`.
    As the default log level of `RootLogger` is `WARNING`, may need to call
    `logging.basicConfig(stream=sys.stdout, level=logging.INFO)` before running this handler to enable
    the stats logging.

    Default behaviors:
        - When EPOCH_COMPLETED, logs ``engine.state.metrics`` using ``self.logger``.
        - When ITERATION_COMPLETED, logs
          ``self.output_transform(engine.state.output)`` using ``self.logger``.

    Usage example::

        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

        trainer = SupervisedTrainer(...)
        StatsHandler(name="train_stats").attach(trainer)

        trainer.run()

    More details of example is available in the tutorial:
    https://github.com/Project-MONAI/tutorials/blob/master/modules/engines/unet_training_dict.py.

    """

    def __init__(
        self,
        iteration_log: bool = True,
        epoch_log: bool = True,
        epoch_print_logger: Optional[Callable[[Engine], Any]] = None,
        iteration_print_logger: Optional[Callable[[Engine], Any]] = None,
        output_transform: Callable = lambda x: x[0],
        global_epoch_transform: Callable = lambda x: x,
        state_attributes: Optional[Sequence[str]] = None,
        name: Optional[str] = None,
        tag_name: str = DEFAULT_TAG,
        key_var_format: str = DEFAULT_KEY_VAL_FORMAT,
    ) -> None:
        """

        Args:
            iteration_log: whether to log data when iteration completed, default to `True`.
            epoch_log: whether to log data when epoch completed, default to `True`.
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
        self.logger = logging.getLogger(name)  # if `name` is None, will default to `engine.logger` when attached
        self.name = name

    def attach(self, engine: Engine) -> None:
        """
        Register a set of Ignite Event-Handlers to a specified Ignite engine.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.

        """
        if self.name is None:
            self.logger = engine.logger
        if self.logger.getEffectiveLevel() > logging.INFO or logging.root.getEffectiveLevel() > logging.INFO:
            warnings.warn(
                "the effective log level of engine logger or RootLogger is higher than INFO, may not record log,"
                " please call `logging.basicConfig(stream=sys.stdout, level=logging.INFO)` to enable it."
            )
        if self.iteration_log and not engine.has_event_handler(self.iteration_completed, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        if self.epoch_log and not engine.has_event_handler(self.epoch_completed, Events.EPOCH_COMPLETED):
            engine.add_event_handler(Events.EPOCH_COMPLETED, self.epoch_completed)
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
        ):
            out_str = f"Key metric: {engine.state.key_metric_name} "  # type: ignore
            out_str += f"best value: {engine.state.best_metric} "  # type: ignore
            out_str += f"at epoch: {engine.state.best_metric_epoch}"  # type: ignore
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
