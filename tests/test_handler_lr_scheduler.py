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

import logging
import sys
import unittest

import numpy as np
import torch
from ignite.engine import Engine, Events

from monai.handlers import LrScheduleHandler


class TestHandlerLrSchedule(unittest.TestCase):
    def test_content(self):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        data = [0] * 8

        # set up engine
        def _train_func(engine, batch):
            pass

        val_engine = Engine(_train_func)
        train_engine = Engine(_train_func)

        @train_engine.on(Events.EPOCH_COMPLETED)
        def run_validation(engine):
            val_engine.run(data)
            val_engine.state.metrics["val_loss"] = 1

        # set up testing handler
        net = torch.nn.PReLU()

        def _reduce_lr_on_plateau():
            optimizer = torch.optim.SGD(net.parameters(), 0.1)
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1)
            handler = LrScheduleHandler(lr_scheduler, step_transform=lambda x: val_engine.state.metrics["val_loss"])
            handler.attach(train_engine)
            return lr_scheduler

        def _reduce_on_step():
            optimizer = torch.optim.SGD(net.parameters(), 0.1)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
            handler = LrScheduleHandler(lr_scheduler)
            handler.attach(train_engine)
            return lr_scheduler

        schedulers = _reduce_lr_on_plateau(), _reduce_on_step()

        train_engine.run(data, max_epochs=5)
        for scheduler in schedulers:
            np.testing.assert_allclose(scheduler._last_lr[0], 0.001)


if __name__ == "__main__":
    unittest.main()
