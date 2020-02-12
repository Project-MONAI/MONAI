import traceback
import logging
from ignite.engine import Engine, Events


class StatsHandler(object):
    """StatsHandler defines a set of Ignite Event-Handlers for all the log printing logics.
    It's can be used for any Ignite Engine(trainer, validator and evaluator).
    And it can support logging for epoch level and iteration level with pre-defined StatsLoggers.

    Args:
        epoch_print_logger (Callable): customized callable printer for epoch level logging.
                                          must accept parameter "engine", use default printer if None.
        iteration_print_logger (Callable): custimized callable printer for iteration level logging.
                                         must accept parameter "engine", use default printer if None.

    """
    def __init__(self,
                 epoch_print_logger=None,
                 iteration_print_logger=None,
                 extra_handler=None):
        self.epoch_print_logger = epoch_print_logger
        self.iteration_print_logger = iteration_print_logger
        self.logger_id = 'stat_logger'
        self.logger = logging.getLogger(logger_id)
        self.logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        self.logger.addHandler(stream_handler)
        if extra_handler is not None:
            self.logger.addHandler(extra_handler)

    def attach(self, engine: Engine):
        """Register a set of Ignite Event-Handlers to a specified Ignite engine.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.

        """
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.epoch_completed)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_started)
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
            self._default_ieration_print(engine)

    def exception_raised(self, engine: Engine, e):
        """handler for train or validation/evaluation exception raised Event.
        Print the exception information and traceback.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.
            e (Exception): the exception catched in Ignite during engine.run().

        """
        self.logger.exception('Exception: {}'.format(e))
        traceback.print_exc()

    def _default_epoch_print(self, engine: Engine):
        """Execute epoch level log operation based on Ignite engine.state data.
        print the values from ignite state.metrics dict.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.

        """
        current_epoch = engine.state.epoch
        num_epochs = engine.state.max_epochs
        prints_dict = engine.state.metrics

        if prints_dict is not None:
            out_str = "Metrics of this epoch {}/{}:  ".format(current_epoch, num_epochs)
            for name in sorted(prints_dict.keys()):
                value = prints_dict[name]
                out_str += "{}: {:.4f}  ".format(name, value)

            self.logger.info(out_str)

    def _default_ieration_print(self, engine: Engine):
        """Execute iteration log operation based on Ignite engine.state data.
        print the values from ignite state.logs dict.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.

        """
        num_iterations = engine.state.epoch_length
        current_iteration = (engine.state.iteration - 1) % num_iterations + 1
        current_epoch = engine.state.epoch
        num_epochs = engine.state.max_epochs
        prints_dict = engine.state.logs

        bar_length = 20
        filled_length = int(bar_length * (current_iteration) // num_iterations)
        bar = '[' + '=' * filled_length + ' ' * (bar_length - filled_length) + ']'
        out_str = "Epoch: {}/{}, Iter: {}/{} {:s}  ".format(
            current_epoch,
            num_epochs,
            current_iteration,
            num_iterations, bar)

        if prints_dict is not None:
            for name in sorted(prints_dict.keys()):
                value = prints_dict[name]
                out_str += "{}: {:.4f}  ".format(name, value)

        self.logger.info(out_str)
