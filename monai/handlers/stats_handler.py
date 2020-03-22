# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
import logging
import torch
from ignite.engine import Engine, Events
from monai.utils.misc import is_scalar

DEFAULT_KEY_VAL_FORMAT = '{}: {:.4f} '
DEFAULT_TAG = 'Loss'


class StatsHandler(object):
    """StatsHandler defines a set of Ignite Event-handlers for all the log printing logics.
    It's can be used for any Ignite Engine(trainer, validator and evaluator).
    And it can support logging for epoch level and iteration level with pre-defined loggers.

    Default behaviors:
        - When EPOCH_COMPLETED, logs ``engine.state.metrics`` using ``self.logger``.
        - When ITERATION_COMPELTED, logs
          ``self.output_transform(engine.state.output)`` using ``self.logger``.

    """

    def __init__(self,
                 epoch_print_logger=None,
                 iteration_print_logger=None,
                 output_transform=lambda x: x,
                 global_epoch_transform=lambda x: x,
                 name=None,
                 tag_name=DEFAULT_TAG,
                 key_var_format=DEFAULT_KEY_VAL_FORMAT):
        """

        Args:
            epoch_print_logger (Callable): customized callable printer for epoch level logging.
                must accept parameter "engine", use default printer if None.
            iteration_print_logger (Callable): custimized callable printer for iteration level logging.
                must accept parameter "engine", use default printer if None.
            output_transform (Callable): a callable that is used to transform the
                ``ignite.engine.output`` into a scalar to print, or a dictionary of {key: scalar}.
                in the latter case, the output string will be formated as key: value.
                by default this value logging happens when every iteration completed.
            global_epoch_transform (Callable): a callable that is used to customize global epoch number.
                For example, in evaluation, the evaluator engine might want to print synced epoch number
                with the trainer engine.
            name (str): identifier of logging.logger to use, defaulting to ``engine.logger``.
            tag_name (string): when iteration output is a scalar, tag_name is used to print
                tag_name: scalar_value to logger. Defaults to ``'Loss'``.
            key_var_format (string): a formatting string to control the output string format of key: value.
        """

        self.epoch_print_logger = epoch_print_logger
        self.iteration_print_logger = iteration_print_logger
        self.output_transform = output_transform
        self.global_epoch_transform = global_epoch_transform
        self.logger = None if name is None else logging.getLogger(name)

        self.tag_name = tag_name
        self.key_var_format = key_var_format

    def attach(self, engine: Engine):
        """Register a set of Ignite Event-Handlers to a specified Ignite engine.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.

        """
        if self.logger is None:
            self.logger = engine.logger
        if not engine.has_event_handler(self.iteration_completed, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        if not engine.has_event_handler(self.epoch_completed, Events.EPOCH_COMPLETED):
            engine.add_event_handler(Events.EPOCH_COMPLETED, self.epoch_completed)
        if not engine.has_event_handler(self.exception_raised, Events.EXCEPTION_RAISED):
            engine.add_event_handler(Events.EXCEPTION_RAISED, self.exception_raised)

    def epoch_completed(self, engine: Engine):
        """handler for train or validation/evaluation epoch completed Event.
        Print epoch level log, default values are from ignite state.metrics dict.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.

        """
        if self.epoch_print_logger is not None:
            self.epoch_print_logger(engine)
        else:
            self._default_epoch_print(engine)

    def iteration_completed(self, engine: Engine):
        """handler for train or validation/evaluation iteration completed Event.
        Print iteration level log, default values are from ignite state.logs dict.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.

        """
        if self.iteration_print_logger is not None:
            self.iteration_print_logger(engine)
        else:
            self._default_iteration_print(engine)

    def exception_raised(self, engine: Engine, e):
        """handler for train or validation/evaluation exception raised Event.
        Print the exception information and traceback.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.
            e (Exception): the exception catched in Ignite during engine.run().

        """
        self.logger.exception('Exception: {}'.format(e))
        # import traceback
        # traceback.print_exc()

    def _default_epoch_print(self, engine: Engine):
        """Execute epoch level log operation based on Ignite engine.state data.
        print the values from ignite state.metrics dict.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.

        """
        prints_dict = engine.state.metrics
        if not prints_dict:
            return
        current_epoch = self.global_epoch_transform(engine.state.epoch)

        out_str = "Epoch[{}] Metrics -- ".format(current_epoch)
        for name in sorted(prints_dict):
            value = prints_dict[name]
            out_str += self.key_var_format.format(name, value)

        self.logger.info(out_str)

    def _default_iteration_print(self, engine: Engine):
        """Execute iteration log operation based on Ignite engine.state data.
        Print the values from ignite state.logs dict.
        Default behaivor is to print loss from output[1], skip if output[1] is not loss.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.

        """
        loss = self.output_transform(engine.state.output)
        if loss is None:
            return  # no printing if the output is empty

        out_str = ''
        if isinstance(loss, dict):  # print dictionary items
            for name in sorted(loss):
                value = loss[name]
                if not is_scalar(value):
                    warnings.warn('ignoring non-scalar output in StatsHandler,'
                                  ' make sure `output_transform(engine.state.output)` returns'
                                  ' a scalar or dictionary of key and scalar pairs to avoid this warning.'
                                  ' {}:{}'.format(name, type(value)))
                    continue  # not printing multi dimensional output
                out_str += self.key_var_format.format(name, value.item() if torch.is_tensor(value) else value)
        else:
            if is_scalar(loss):  # not printing multi dimensional output
                out_str += self.key_var_format.format(self.tag_name, loss.item() if torch.is_tensor(loss) else loss)
            else:
                warnings.warn('ignoring non-scalar output in StatsHandler,'
                              ' make sure `output_transform(engine.state.output)` returns'
                              ' a scalar or a dictionary of key and scalar pairs to avoid this warning.'
                              ' {}'.format(type(loss)))

        if not out_str:
            return  # no value to print

        num_iterations = engine.state.epoch_length
        current_iteration = (engine.state.iteration - 1) % num_iterations + 1
        current_epoch = engine.state.epoch
        num_epochs = engine.state.max_epochs

        base_str = "Epoch: {}/{}, Iter: {}/{} --".format(
            current_epoch,
            num_epochs,
            current_iteration,
            num_iterations)

        self.logger.info(' '.join([base_str, out_str]))
