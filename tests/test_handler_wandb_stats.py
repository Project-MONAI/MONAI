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

import torch
import unittest
from concurrent.futures import ThreadPoolExecutor

from ignite.engine import Engine

from monai.handlers import WandbStatsHandler
from monai.utils import optional_import

wandb, _ = optional_import("wandb")


def dummy_train(start):
    # set up engine
    def _train_func(engine, batch):
        return batch + 1.0

    engine = Engine(_train_func)

    # set up testing handler
    handler = WandbStatsHandler(
        output_transform=lambda x: x,
    )
    handler.attach(engine)
    engine.run(torch.tensor([start]), max_epochs=5)
    handler.close()


@unittest.skipUnless(wandb, "Requires wandb installation")
class TestHandlerWB(unittest.TestCase):
    def test_multi_thread(self):
        wandb.init(
            project="multithread-handlers", save_code=True, sync_tensorboard=True
        )
        with ThreadPoolExecutor(2, "Training") as executor:
            for t in range(2):
                executor.submit(dummy_train, t + 2)
