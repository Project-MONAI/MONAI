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

import numpy as np
import torch
import torch.distributed as dist

from monai.metrics import CumulativeAverage
from tests.utils import SkipIfBeforePyTorchVersion, skip_if_no_cuda


def main_worker(rank, nprocs):

    is_cuda = torch.cuda.is_available()

    if nprocs > 1 and is_cuda:
        dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:12345", world_size=nprocs, rank=rank)
        torch.cuda.set_device(rank)
        device = torch.device(rank)
    else:
        device = torch.device("cpu")

    avg_meter = CumulativeAverage()  # each process rank has it's own AverageMeter
    n_iter = 10
    for i in range(n_iter):
        val = torch.as_tensor(rank + i, device=device)
        avg_meter.append(val=val)

    avg_val = avg_meter.aggregate()  # average across all processes
    expected_val = sum(sum(list(range(rank_i, rank_i + n_iter))) for rank_i in range(nprocs)) / (n_iter * nprocs)
    np.testing.assert_equal(avg_val, expected_val)

    if dist.is_initialized():
        dist.destroy_process_group()


@skip_if_no_cuda
@SkipIfBeforePyTorchVersion((1, 8))
class DistributedCumulativeAverage2(unittest.TestCase):
    def test_value(self):
        ngpus_per_node = torch.cuda.device_count()
        if ngpus_per_node > 1:
            torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node,))
        else:
            main_worker(0, 1)


if __name__ == "__main__":
    unittest.main()
