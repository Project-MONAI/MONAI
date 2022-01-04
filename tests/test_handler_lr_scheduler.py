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

import logging
import os
import re
import sys
import tempfile
import unittest

import numpy as np
import torch
from ignite.engine import Engine, Events

from monai.handlers import LrScheduleHandler


class TestHandlerLrSchedule(unittest.TestCase):
    def test_content(self):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        data = [0] * 8
        test_lr = 0.1
        gamma = 0.1

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
            optimizer = torch.optim.SGD(net.parameters(), test_lr)
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1)
            handler = LrScheduleHandler(lr_scheduler, step_transform=lambda x: val_engine.state.metrics["val_loss"])
            handler.attach(train_engine)
            return handler

        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test_lr.log")
            # test with additional logging handler
            file_saver = logging.FileHandler(filename, mode="w")
            file_saver.setLevel(logging.INFO)

            def _reduce_on_step():
                optimizer = torch.optim.SGD(net.parameters(), test_lr)
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=gamma)
                handler = LrScheduleHandler(lr_scheduler, name="test_logging", logger_handler=file_saver)
                handler.attach(train_engine)
                return handler

            schedulers = _reduce_lr_on_plateau(), _reduce_on_step()

            train_engine.run(data, max_epochs=5)
            file_saver.close()
            schedulers[1].logger.removeHandler(file_saver)

            with open(filename) as f:
                output_str = f.read()
                grep = re.compile(".*Current learning rate.*")
                content_count = 0
                for line in output_str.split("\n"):
                    if grep.match(line):
                        content_count += 1
                self.assertTrue(content_count > 0)

        for scheduler in schedulers:
            np.testing.assert_allclose(scheduler.lr_scheduler._last_lr[0], 0.001)


if __name__ == "__main__":
    unittest.main()
