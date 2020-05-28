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

import os
import tempfile
import shutil
import torch
import unittest
from ignite.engine import Engine
from monai.handlers import CheckpointSaver
import torch.optim as optim
from parameterized import parameterized
import logging
import sys

TEST_CASE_1 = [True, False, None, 1, True, 0, None, ["test_checkpoint_final_iteration=40.pth"]]

TEST_CASE_2 = [
    False,
    True,
    "val_loss",
    2,
    True,
    0,
    None,
    ["test_checkpoint_key_metric=32.pth", "test_checkpoint_key_metric=40.pth"],
]

TEST_CASE_3 = [False, False, None, 1, True, 2, 2, ["test_checkpoint_epoch=2.pth", "test_checkpoint_epoch=4.pth"]]

TEST_CASE_4 = [
    False,
    False,
    None,
    1,
    False,
    10,
    2,
    ["test_checkpoint_iteration=30.pth", "test_checkpoint_iteration=40.pth"],
]

TEST_CASE_5 = [True, False, None, 1, True, 0, None, ["test_checkpoint_final_iteration=40.pth"], True]


class TestHandlerCheckpointSaver(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5])
    def test_file(
        self,
        save_final,
        save_key_metric,
        key_metric_name,
        key_metric_n_saved,
        epoch_level,
        save_interval,
        n_saved,
        filenames,
        multi_devices=False,
    ):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        data = [0] * 8

        # set up engine
        def _train_func(engine, batch):
            engine.state.metrics["val_loss"] = engine.state.iteration

        engine = Engine(_train_func)

        # set up testing handler
        net = torch.nn.PReLU()
        if multi_devices:
            net = torch.nn.DataParallel(net)
        optimizer = optim.SGD(net.parameters(), lr=0.02)
        tempdir = tempfile.mkdtemp()
        handler = CheckpointSaver(
            tempdir,
            {"net": net, "opt": optimizer},
            "CheckpointSaver",
            "test",
            save_final,
            save_key_metric,
            key_metric_name,
            key_metric_n_saved,
            epoch_level,
            save_interval,
            n_saved,
        )
        handler.attach(engine)
        engine.run(data, max_epochs=5)
        for filename in filenames:
            self.assertTrue(os.path.exists(os.path.join(tempdir, filename)))
        shutil.rmtree(tempdir)


if __name__ == "__main__":
    unittest.main()
