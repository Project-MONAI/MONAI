from ignite.engine import Engine, Events


class TrainHandler(object):
    def __init__(self):
        self.callbacks = {
            Events.STARTED: self.started_callback,
            Events.COMPLETED: self.completed_callback,
            Events.EPOCH_STARTED: self.epoch_started_callback,
            Events.EPOCH_COMPLETED: self.epoch_completed_callback,
            Events.ITERATION_STARTED: self.step_started_callback,
            Events.ITERATION_COMPLETED: self.step_completed_callback,
            Events.EXCEPTION_RAISED: self.exception_raised_callback
        }

    def start(self, engine: Engine):
        for event, callback in self.callbacks.items():
            engine.add_event_handler(event_name=event, handler=callback)

    def get_callbacks(self):
        return self.callbacks

    def started_callback(self, engine):
        # default callback action
        pass

    def completed_callback(self, engine):
        # default callback action
        pass

    def epoch_started_callback(self, engine):
        # default callback action
        pass

    def epoch_completed_callback(self, engine):
        # default callback action
        pass

    def step_started_callback(self, engine):
        # default callback action
        pass

    def step_completed_callback(self, engine):
        # default callback action
        pass

    def exception_raised_callback(self, engine, e):
        # default callback action
        pass
