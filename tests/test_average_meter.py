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
from parameterized import parameterized

from monai.metrics import AverageMeter

# from tests.utils import assert_allclose

# single class value
TEST_CASE_1 = []
TEST_CASE_1.append([{"vals": [1, 2, 3], "avg": 2}])
TEST_CASE_1.append([{"vals": [[1, 1, 1], [2, 2, 2], [3, 6, 9]], "avg": [2, 3, 4]}])

TEST_CASE_1.append([{"vals": [2, 4, 6], "counts": [2, 1, 2], "avg": 4}])
TEST_CASE_1.append(
    [{"vals": [[3, 2, 1], [2, 3, 2], [0, 0, 9]], "counts": [[4, 4, 4], [4, 4, 4], [2, 2, 2]], "avg": [2, 2, 3]}]
)

TEST_CASE_1.append([{"vals": [1, 2, float("nan")], "avg": 1.5}])


class TestAverageMeter(unittest.TestCase):
    @parameterized.expand(TEST_CASE_1)
    def test_value_all(self, data):

        # test orig
        self.run_test(data)

        # test in numpy
        data["vals"] = np.array(data["vals"])
        data["avg"] = np.array(data["avg"])
        self.run_test(data)

        # test as Tensors
        data["vals"] = torch.tensor(data["vals"])
        data["avg"] = torch.tensor(data["avg"], dtype=torch.float)
        self.run_test(data)

    def run_test(self, data):
        vals = data["vals"]
        avg = data["avg"]

        counts = data.get("counts", None)
        if counts is not None and not isinstance(counts, list) and isinstance(vals, list):
            counts = [counts] * len(vals)

        avg_meter = AverageMeter()
        for i in range(len(vals)):
            if counts is not None:
                avg_meter.update(vals[i], counts[i])
            else:
                avg_meter.update(vals[i])

        np.testing.assert_equal(avg_meter.get_avg(), avg)


def main_worker(rank, nprocs):

    has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 1
    backend = "nccl" if has_cuda else "gloo"

    dist.init_process_group(backend=backend, init_method="tcp://127.0.0.1:12345", world_size=nprocs, rank=rank)

    avg_meter = AverageMeter()  # each process rank has it's own AverageMeter
    n_iter = 10
    for i in range(n_iter):
        val = rank + i
        avg_meter.update(val=val)

    avg_val = avg_meter.get_avg()  # average across all processes
    expected_val = sum(sum(list(range(rank_i, rank_i + n_iter))) for rank_i in range(nprocs)) / (n_iter * nprocs)
    np.testing.assert_equal(avg_val, expected_val)

    if dist.is_initialized():
        dist.destroy_process_group()


class TestDDP(unittest.TestCase):
    def test_ddp_ops(self):
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            nprocs = torch.cuda.device_count()
        else:
            nprocs = 2

        torch.multiprocessing.spawn(main_worker, nprocs=nprocs, args=(nprocs,))


if __name__ == "__main__":
    unittest.main()
