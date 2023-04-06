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

from __future__ import annotations

import os
import glob
import shutil
import tempfile
import unittest

from ignite.engine import Engine, Events

from monai.utils import optional_import

wandb, _ = optional_import("wandb")

from monai.networks.nets import UNet
from monai.handlers import WandbModelCheckpointHandler


@unittest.skipUnless(wandb, "Requires wandb installation")
class TestWandbModelCheckpointHandler(unittest.TestCase):
    def test_model_checkpointing(self):
        tempdir = tempfile.TemporaryDirectory()
        os.system("wandb offline")
        os.environ["WANDB_DIR"] = tempdir.name

        wandb.init()

        net = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

        # set up engine
        def _train_func(engine, batch):
            return [batch + 1.0]

        engine = Engine(_train_func)

        handler = WandbModelCheckpointHandler(dirname=tempdir.name, filename_prefix="test", save_interval=1)
        engine.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=handler, to_save={"net": net})

        engine.run(range(3), max_epochs=2)

        self.assertTrue(os.path.isdir(tempdir.name))
        self.assertTrue(len(glob.glob(os.path.join(tempdir.name, "*"))) > 0)
        self.assertTrue(len(glob.glob(os.path.join(tempdir.name, "wandb", "*"))) > 0)
        self.assertTrue(len(glob.glob(os.path.join(tempdir.name, "wandb", "debug*"))) > 0)
        self.assertTrue(os.path.isfile(handler.last_checkpoint))

        shutil.rmtree(tempdir.name)
