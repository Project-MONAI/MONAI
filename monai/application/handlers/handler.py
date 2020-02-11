from ignite.engine import Engine, Events


class Handler(object):
    """Handler defines a set of Ignite Event-Handlers and acts as base class.
    There are 9 pre-defined Events in Ignite engine for trainer, validator and evaluator.
    So users can define 1 set of Event-Handlers for 1 specific usage.
    And they must be compatible with trainer, validator and evaluator, like:
    StatsHandler for all the log printing and TensorBoard summary related logics at 9 different Events.

    """
    def __init__(self):
        self.callbacks = {
            Events.STARTED: self.started_callback,
            Events.COMPLETED: self.completed_callback,
            Events.EPOCH_STARTED: self.epoch_started_callback,
            Events.EPOCH_COMPLETED: self.epoch_completed_callback,
            Events.GET_BATCH_STARTED: self.get_batch_started_callback,
            Events.GET_BATCH_COMPLETED: self.get_batch_completed_callback,
            Events.ITERATION_STARTED: self.iteration_started_callback,
            Events.ITERATION_COMPLETED: self.iteration_completed_callback,
            Events.EXCEPTION_RAISED: self.exception_raised_callback
        }

    def attach(self, engine: Engine):
        """Register a set of Ignite Event-Handlers to a specified Ignite engine.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.

        """
        for event, callback in self.callbacks.items():
            engine.add_event_handler(event_name=event, handler=callback)

    def get_callbacks(self):
        """Util function to return the callbacks for all 7 Ignite Events.

        """
        return self.callbacks

    def started_callback(self, engine):
        """Callback for train or validation/evaluation started Event.
        No default action in base class, so subclass can ignore to extend if no action at started Event.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.

        """
        pass

    def completed_callback(self, engine):
        """Callback for train or validation/evaluation completed Event.
        No default action in base class, so subclass can ignore to extend if no action at completed Event.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.

        """
        pass

    def epoch_started_callback(self, engine):
        """Callback for train or validation/evaluation epoch started Event.
        No default action in base class, so subclass can ignore to extend if no action at epoch started Event.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.

        """
        pass

    def epoch_completed_callback(self, engine):
        """Callback for train or validation/evaluation epoch completed Event.
        No default action in base class, so subclass can ignore to extend if no action at epoch completed Event.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.

        """
        pass

    def get_batch_started_callback(self, engine):
        """Callback for train or validation/evaluation get batched data started Event.
        No default action in base class, so subclass can ignore to extend if no action at get batch started Event.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.

        """
        pass

    def get_batch_completed_callback(self, engine):
        """Callback for train or validation/evaluation get batched data completed Event.
        No default action in base class, so subclass can ignore to extend if no action at get batch completed Event.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.

        """
        pass

    def iteration_started_callback(self, engine):
        """Callback for train or validation/evaluation iteration started Event.
        No default action in base class, so subclass can ignore to extend if no action at iteration started Event.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.

        """
        pass

    def iteration_completed_callback(self, engine):
        """Callback for train or validation/evaluation iteration completed Event.
        No default action in base class, so subclass can ignore to extend if no action at iteration completed Event.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.

        """
        pass

    def exception_raised_callback(self, engine, e):
        """Callback for train or validation/evaluation exception raised Event.
        No default action in base class, so subclass can ignore to extend if no action at exception raised Event.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.
            e (Exception): the exception catched in Ignite during engine.run().

        """
        pass
