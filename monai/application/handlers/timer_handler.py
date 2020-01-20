import time
from .train_handler import TrainHandler


class TimerHandler(TrainHandler):
    def __init__(self):
        TrainHandler.__init__(self)

    def started_callback(self, engine):
        engine.state.train_time = {'TRAIN_START': time.time()}

    def completed_callback(self, engine):
        engine.state.train_time['TRAIN_END'] = time.time()

    def epoch_started_callback(self, engine):
        engine.state.train_time['EPOCH_START'] = time.time()

    def epoch_completed_callback(self, engine):
        engine.state.train_time['EPOCH_END'] = time.time()

    def step_started_callback(self, engine):
        engine.state.train_time['STEP_START'] = time.time()

    def step_completed_callback(self, engine):
        engine.state.train_time['STEP_END'] = time.time()

    def exception_raised_callback(self, engine, e):
        engine.state.train_time['TRAIN_END'] = time.time()
