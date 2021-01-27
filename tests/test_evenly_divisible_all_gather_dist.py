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

import unittest

import torch
import torch.distributed as dist

from monai.handlers import evenly_divisible_all_gather
from tests.utils import DistCall, DistTestCase


class DistributedEvenlyDivisibleAllGather(DistTestCase):
    @DistCall(nnodes=1, nproc_per_node=2)
    def test_data(self):
        self._run()

    def _run(self):
        if dist.get_rank() == 0:
            data1 = torch.tensor([[1, 2], [3, 4]])
            data2 = torch.tensor([[1.0, 2.0]])

        if dist.get_rank() == 1:
            data1 = torch.tensor([[5, 6]])
            data2 = torch.tensor([[3.0, 4.0], [5.0, 6.0]])

        result1 = evenly_divisible_all_gather(data=data1)
        torch.testing.assert_allclose(result1, torch.tensor([[1, 2], [3, 4], [5, 6]]))
        result2 = evenly_divisible_all_gather(data=data2)
        torch.testing.assert_allclose(result2, torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))


if __name__ == "__main__":
    unittest.main()
