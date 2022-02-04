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
import torch.distributed as dist

from monai.engines import create_multigpu_supervised_trainer
from tests.utils import DistCall, DistTestCase, skip_if_no_cuda


def fake_loss(y_pred, y):
    return (y_pred[0] + y).sum()


def fake_data_stream():
    while True:
        yield torch.rand((10, 1, 64, 64)), torch.rand((10, 1, 64, 64))


class DistributedTestParallelExecution(DistTestCase):
    @DistCall(nnodes=1, nproc_per_node=2)
    @skip_if_no_cuda
    def test_distributed(self):
        device = torch.device(f"cuda:{dist.get_rank()}")
        net = torch.nn.Conv2d(1, 1, 3, padding=1).to(device)
        opt = torch.optim.Adam(net.parameters(), 1e-3)

        trainer = create_multigpu_supervised_trainer(net, opt, fake_loss, [device], distributed=True)
        trainer.run(fake_data_stream(), 2, 2)
        # assert the trainer output is loss value
        self.assertTrue(isinstance(trainer.state.output, float))


if __name__ == "__main__":
    unittest.main()
