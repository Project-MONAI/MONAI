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

import logging

from ignite.engine import Engine, Events

KEY_VAL_FORMAT = '{}: {:.4f} '


class StatsHandler(object):
    """StatsHandler defines a set of Ignite Event-handlers for all the log printing logics.
    It's can be used for any Ignite Engine(trainer, validator and evaluator).
    And it can support logging for epoch level and iteration level with pre-defined StatsLoggers.
    By default, this class logs the dictionary of `engine.state.metrics`.
    """

    def __init__(self,
                 epoch_print_logger=None,
                 iteration_print_logger=None,
                 name=None):
        """
        Args:
            epoch_print_logger (Callable): customized callable printer for epoch level logging.
                                            must accept parameter "engine", use default printer if None.
            iteration_print_logger (Callable): custimized callable printer for iteration level logging.
                                            must accept parameter "engine", use default printer if None.
            name (str): identifier of logging.logger to use, defaulting to `engine.logger`.
        """

        self.epoch_print_logger = epoch_print_logger
        self.iteration_print_logger = iteration_print_logger

        self.logger = None if name is None else logging.getLogger(name)


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
        current_epoch = engine.state.epoch

        out_str = "Epoch[{}] Metrics -- ".format(current_epoch)
        for name in sorted(prints_dict):
            value = prints_dict[name]
            out_str += KEY_VAL_FORMAT.format(name, value)

        self.logger.info(out_str)

    def _default_iteration_print(self, engine: Engine):
        """Execute iteration log operation based on Ignite engine.state data.
        print the values from ignite state.logs dict.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.

        """
        prints_dict = engine.state.metrics
        if not prints_dict:
            return
        num_iterations = engine.state.epoch_length
        current_iteration = (engine.state.iteration - 1) % num_iterations + 1
        current_epoch = engine.state.epoch
        num_epochs = engine.state.max_epochs

        out_str = "Epoch: {}/{}, Iter: {}/{} -- ".format(
            current_epoch,
            num_epochs,
            current_iteration,
            num_iterations)

        for name in sorted(prints_dict):
            value = prints_dict[name]
            out_str += KEY_VAL_FORMAT.format(name, value)

        self.logger.info(out_str)
