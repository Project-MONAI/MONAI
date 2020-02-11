import traceback
from .handler import Handler


class StatsHandler(Handler):
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
                 iteration_print_logger=None):
        Handler.__init__(self)
        self.epoch_print_logger = epoch_print_logger
        self.iteration_print_logger = iteration_print_logger

    def epoch_completed_callback(self, engine):
        """Callback for train or validation/evaluation epoch completed Event.
        Print epoch level log, default values are from ignite state.metrics dict.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.

        """
        if self.epoch_print_logger is not None:
            self.epoch_print_logger(engine)
        else:
            self._default_epoch_print(engine)

    def iteration_completed_callback(self, engine):
        """Callback for train or validation/evaluation iteration completed Event.
        Print iteration level log, default values are from ignite state.logs dict.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.

        """
        if self.iteration_print_logger is not None:
            self.iteration_print_logger(engine)
        else:
            self._default_ieration_print(engine)

    def exception_raised_callback(self, engine, e):
        """Callback for train or validation/evaluation exception raised Event.
        Print the exception information and traceback.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.
            e (Exception): the exception catched in Ignite during engine.run().

        """
        print('Exception:', e)
        traceback.print_exc()

    def _default_epoch_print(self, engine):
        """Execute epoch level log operation based on Ignite engine.state data.
        print the values from ignite state.metrics dict.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.

        """
        current_epoch = engine.state.epoch
        num_epochs = engine.state.max_epochs
        prints_dict = engine.state.metrics

        if prints_dict is not None:
            print('Metrics of this epoch {}/{}:'.format(current_epoch, num_epochs))
            for name in sorted(prints_dict.keys()):
                value = prints_dict[name]
                print('{}: {:.4f}  '.format(name, value), end='')

    def _default_ieration_print(self, engine):
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

        bar_length = self.bar_length
        filled_length = int(bar_length * (current_iteration) // num_iterations)
        bar = '[' + '=' * filled_length + ' ' * (bar_length - filled_length) + ']'
        print("Epoch: {}/{}, Iter: {}/{} {:s}  ".format(
            current_epoch,
            num_epochs,
            current_iteration,
            num_iterations, bar), end='')

        if prints_dict is not None:
            for name in sorted(prints_dict.keys()):
                value = prints_dict[name]
                print("{}: {:.4f}  ".format(name, value), end='')

        if current_iteration >= num_iterations and not self.new_line:
            # new line
            print("")
