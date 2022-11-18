# Copyright (c) MONAI Consortium
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

from monai.data import Dataset
from monai.engines import Evaluator
from monai.handlers import ValidationHandler


class TestEvaluator(Evaluator):
    def _iteration(self, engine, batchdata):
        pass


class TestHandlerValidation(unittest.TestCase):
    def test_content(self):
        data = [0] * 8

        # set up engine
        def _train_func(engine, batch):
            pass

        engine = Engine(_train_func)

        # set up testing handler
        val_data_loader = torch.utils.data.DataLoader(Dataset(data))
        evaluator = TestEvaluator(torch.device("cpu:0"), val_data_loader)
        saver = ValidationHandler(interval=2, validator=evaluator)
        saver.attach(engine)

        engine.run(data, max_epochs=5)
        self.assertEqual(evaluator.state.max_epochs, 4)
        self.assertEqual(evaluator.state.epoch_length, 8)


if __name__ == "__main__":
    unittest.main()
