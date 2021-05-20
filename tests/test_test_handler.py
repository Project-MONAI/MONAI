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

import torch
from ignite.engine import Engine

from monai.handlers.handler import TestHandler
from monai.handlers.utils import attach_ignite_engine


class TestTestHandler(unittest.TestCase):
    def test_test_handler(self):
        # set up engine
        def _train_func(engine, batch):
            return torch.tensor(0.0)

        engine = Engine(_train_func)

        testhandler = TestHandler()
        attach_ignite_engine(engine, testhandler)
        engine.run(range(3), max_epochs=2)


if __name__ == "__main__":
    unittest.main()
