import time
from .train_handler import TrainHandler


class TimerHandler(TrainHandler):
    def __init__(self):
        TrainHandler.__init__(self)

    def started_callback(self, engine):
        engine.state.metrics['TRAIN_START_TIME'] = time.time()

    def completed_callback(self, engine):
        engine.state.metrics['TRAIN_END_TIME'] = time.time()

    def epoch_started_callback(self, engine):
        engine.state.metrics['EPOCH_START_TIME'] = time.time()

    def epoch_completed_callback(self, engine):
        engine.state.metrics['EPOCH_END_TIME'] = time.time()

    def step_started_callback(self, engine):
        engine.state.metrics['STEP_START_TIME'] = time.time()

    def step_completed_callback(self, engine):
        engine.state.metrics['STEP_END_TIME'] = time.time()

    def exception_raised_callback(self, engine, e):
        engine.state.metrics['TRAIN_END_TIME'] = time.time()
