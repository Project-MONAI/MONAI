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

import time
import unittest

import torch

from monai.data import DataLoader
from monai.engines import CommonKeys, SupervisedTrainer
from monai.utils import ThreadContainer
from tests.utils import skip_if_quick


class TestThreadContainer(unittest.TestCase):
    @skip_if_quick
    def test_container(self):
        net = torch.nn.Conv2d(1, 1, 3, padding=1)

        opt = torch.optim.Adam(net.parameters())

        img = torch.rand(1, 16, 16)
        data = {CommonKeys.IMAGE: img, CommonKeys.LABEL: img}
        loader = DataLoader([data for _ in range(10)])

        trainer = SupervisedTrainer(
            device=torch.device("cpu"),
            max_epochs=1,
            train_data_loader=loader,
            network=net,
            optimizer=opt,
            loss_function=torch.nn.L1Loss(),
        )

        con = ThreadContainer(trainer)
        con.start()
        time.sleep(1)  # wait for trainer to start

        self.assertTrue(con.is_alive)
        self.assertIsNotNone(con.status())
        self.assertTrue(len(con.status_dict) > 0)

        con.join()
