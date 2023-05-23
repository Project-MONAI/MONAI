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

import glob
import os
import shutil
import tempfile
import unittest

from ignite.engine import Engine

from monai.handlers import WandbStatsHandler
from monai.utils import optional_import

wandb, _ = optional_import("wandb")


@unittest.skipUnless(wandb, "Requires wandb installation")
class TestWandbStatsHandler(unittest.TestCase):
    def test_metric_tracking(self):
        tempdir = tempfile.TemporaryDirectory()
        os.system("wandb offline")
        os.environ["WANDB_DIR"] = tempdir.name

        wandb.init(dir=tempdir.name)

        # set up engine
        def _train_func(engine, batch):
            return [batch + 1.0]

        engine = Engine(_train_func)

        handler = WandbStatsHandler(output_transform=lambda x: x)
        handler.attach(engine)

        engine.run(range(3), max_epochs=2)

        self.assertTrue(os.path.isdir(tempdir.name))
        self.assertTrue(len(glob.glob(os.path.join(tempdir.name, "*"))) > 0)

        wandb.finish()

        shutil.rmtree(tempdir.name)
