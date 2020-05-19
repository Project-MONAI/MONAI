# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import unittest
import numpy as np
from ignite.engine import Engine
from monai.networks.nets import UNet
from monai.handlers import LrScheduleHander


class TestHandlerLrSchedule(unittest.TestCase):
    def test_content(self):
        data = [0] * 8

        # set up engine
        def _train_func(engine, batch):
            pass

        engine = Engine(_train_func)

        # set up testing handler
        net = UNet(
            dimensions=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        optimizer = torch.optim.Adam(net.parameters(), 0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        handler = LrScheduleHander(lr_scheduler)
        handler.attach(engine)

        engine.run(data, max_epochs=5)
        np.testing.assert_allclose(lr_scheduler.get_last_lr()[0], 1e-06, rtol=1e-06)


if __name__ == "__main__":
    unittest.main()
