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
from typing import Dict
import torch
from ignite.engine import Engine

from monai.handlers.handler import Handler
from monai.handlers.utils import attach_ignite_engine
from monai.utils.enums import Events


class DemoHandler(Handler):
    def __init__(self) -> None:
        super().__init__()
        self._register(event=Events.STARTED, func=self._started)
        self._register(event=Events.ITERATION_COMPLETED, func=self._iteration)
        self._register(event=Events.EPOCH_COMPLETED, func=self._epoch)

    def _started(self, data: Dict):
        data["state"]["magic_number"] = data['state']['max_epochs']

    def _iteration(self, data: Dict):
        data["state"]["magic_number"] += data['state']['iteration']

    def _epoch(self, data: Dict):
        data["state"]["output"] = torch.tensor(1.0)


class TestHandler(unittest.TestCase):
    def test_test_handler(self):
        # set up engine
        def _train_func(engine, batch):
            return torch.tensor(0.0)

        engine = Engine(_train_func)

        testhandler = DemoHandler()
        attach_ignite_engine(engine, testhandler)
        engine.run(range(3), max_epochs=2)
        self.assertEqual(engine.state.magic_number, 23)
        self.assertEqual(engine.state.output.item(), 1.0)


if __name__ == "__main__":
    unittest.main()
