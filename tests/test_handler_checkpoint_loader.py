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
import tempfile
import unittest

import torch
import torch.optim as optim
from ignite.engine import Engine

from monai.handlers import CheckpointLoader, CheckpointSaver


class TestHandlerCheckpointLoader(unittest.TestCase):
    def test_one_save_one_load(self):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        net1 = torch.nn.PReLU()
        data1 = net1.state_dict()
        data1["weight"] = torch.tensor([0.1])
        net1.load_state_dict(data1)
        net2 = torch.nn.PReLU()
        data2 = net2.state_dict()
        data2["weight"] = torch.tensor([0.2])
        net2.load_state_dict(data2)
        with tempfile.TemporaryDirectory() as tempdir:
            engine = Engine(lambda e, b: None)
            CheckpointSaver(save_dir=tempdir, save_dict={"net": net1}, save_final=True).attach(engine)
            engine.run([0] * 8, max_epochs=5)
            path = tempdir + "/net_final_iteration=40.pt"
            engine = Engine(lambda e, b: None)
            CheckpointLoader(load_path=path, load_dict={"net": net2}).attach(engine)
            engine.run([0] * 8, max_epochs=1)
            torch.testing.assert_allclose(net2.state_dict()["weight"], 0.1)

    def test_two_save_one_load(self):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        net1 = torch.nn.PReLU()
        optimizer = optim.SGD(net1.parameters(), lr=0.02)
        data1 = net1.state_dict()
        data1["weight"] = torch.tensor([0.1])
        net1.load_state_dict(data1)
        net2 = torch.nn.PReLU()
        data2 = net2.state_dict()
        data2["weight"] = torch.tensor([0.2])
        net2.load_state_dict(data2)
        with tempfile.TemporaryDirectory() as tempdir:
            engine = Engine(lambda e, b: None)
            save_dict = {"net": net1, "opt": optimizer}
            CheckpointSaver(save_dir=tempdir, save_dict=save_dict, save_final=True).attach(engine)
            engine.run([0] * 8, max_epochs=5)
            path = tempdir + "/checkpoint_final_iteration=40.pt"
            engine = Engine(lambda e, b: None)
            CheckpointLoader(load_path=path, load_dict={"net": net2}).attach(engine)
            engine.run([0] * 8, max_epochs=1)
            torch.testing.assert_allclose(net2.state_dict()["weight"], 0.1)

    def test_save_single_device_load_multi_devices(self):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        net1 = torch.nn.PReLU()
        data1 = net1.state_dict()
        data1["weight"] = torch.tensor([0.1])
        net1.load_state_dict(data1)
        net2 = torch.nn.PReLU()
        data2 = net2.state_dict()
        data2["weight"] = torch.tensor([0.2])
        net2.load_state_dict(data2)
        net2 = torch.nn.DataParallel(net2)
        with tempfile.TemporaryDirectory() as tempdir:
            engine = Engine(lambda e, b: None)
            CheckpointSaver(save_dir=tempdir, save_dict={"net": net1}, save_final=True).attach(engine)
            engine.run([0] * 8, max_epochs=5)
            path = tempdir + "/net_final_iteration=40.pt"
            engine = Engine(lambda e, b: None)
            CheckpointLoader(load_path=path, load_dict={"net": net2}).attach(engine)
            engine.run([0] * 8, max_epochs=1)
            torch.testing.assert_allclose(net2.state_dict()["module.weight"].cpu(), 0.1)


if __name__ == "__main__":
    unittest.main()
