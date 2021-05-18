# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from abc import ABC, abstractmethod
from typing import Callable, Dict, Sequence

import torch

from monai.utils import optional_import
from tests.utils import SkipIfNoModule

try:
    _, has_ignite = optional_import("ignite")
    from ignite.engine import Engine, Events, State

    from monai.handlers import MetricLogger
except ImportError:
    has_ignite = False


class DictState(State):
    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class DictEngine(Engine):
    def __init__(self, process_function: Callable):
        super().__init__(process_function)
        self.state = DictState()

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)


def attach_engine(engine: DictEngine, handler: Callable):
    for event in handler.get_events():
        # pass the event as kwarg to handler callback
        engine.add_event_handler(Events[event], handler, event=event)


class EVENTS:
    STARTED = "STARTED"
    ITERATION_COMPLETED = "ITERATION_COMPLETED"
    EPOCH_COMPLETED = "EPOCH_COMPLETED"
    COMPLETED = "COMPLETED"
    EXCEPTION_RAISED = "EXCEPTION_RAISED"


class Handler(ABC):
    """
    Base class of all handlers

    """

    def __init__(self, events: Sequence[str]) -> None:
        self.events = events

    def get_events(self):
        return self.events

    @abstractmethod
    def __call__(self, data: Dict, event: str):
        # data should have the same structure as ignite.Engine,
        # which need to support dict properties
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class TestHandler(Handler):
    def __init__(self) -> None:
        super().__init__(events=[EVENTS.STARTED, EVENTS.ITERATION_COMPLETED, EVENTS.EPOCH_COMPLETED])

    def __call__(self, data: Dict, event: str):
        # data should have the same structure as ignite.Engine,
        # which need to support dict properties
        if event == EVENTS.STARTED:
            self._started(data)
        if event == EVENTS.ITERATION_COMPLETED:
            self._iteration(data)
        if event == EVENTS.EPOCH_COMPLETED:
            self._epoch(data)

    def _started(self, data: Dict):
        print(f"total epochs: {data['state']['max_epochs']}.")

    def _iteration(self, data: Dict):
        print(f"current iteration: {data['state']['iteration']}.")

    def _epoch(self, data: Dict):
        print(f"should terminated: {data['should_terminate']}.")


class TestHandlerMetricLogger(unittest.TestCase):
    @SkipIfNoModule("ignite")
    def test_metric_logging(self):
        dummy_name = "dummy"

        # set up engine
        def _train_func(engine, batch):
            return torch.tensor(0.0)

        engine = Engine(_train_func)

        # set up dummy metric
        @engine.on(Events.EPOCH_COMPLETED)
        def _update_metric(engine):
            engine.state.metrics[dummy_name] = 1

        # set up testing handler
        handler = MetricLogger(loss_transform=lambda output: output.item())
        handler.attach(engine)

        engine.run(range(3), max_epochs=2)

        expected_loss = [(1, 0.0), (2, 0.0), (3, 0.0), (4, 0.0), (5, 0.0), (6, 0.0)]
        expected_metric = [(4, 1), (5, 1), (6, 1)]

        self.assertSetEqual({dummy_name}, set(handler.metrics))

        self.assertListEqual(expected_loss, handler.loss)
        self.assertListEqual(expected_metric, handler.metrics[dummy_name])

        engine = DictEngine(_train_func)
        testhandler = TestHandler()
        attach_engine(engine, testhandler)
        engine.run(range(3), max_epochs=2)


if __name__ == "__main__":
    unittest.main()
