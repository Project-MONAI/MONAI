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

from monai.metrics import CumulativeAverage
from tests.utils import DistCall, DistTestCase, assert_allclose


class DistributedCumulativeAverage(DistTestCase):
    @DistCall(nnodes=1, nproc_per_node=2)
    def test_value(self):
        rank = dist.get_rank()
        input_data = [
            [torch.as_tensor([[0.1]]), torch.as_tensor([[0.2]]), torch.as_tensor([[0.3]])],
            [torch.as_tensor([[0.1]]), torch.as_tensor([[0.2]]), torch.as_tensor([[float("nan")]])],
            [torch.as_tensor([[0.1, 0.2]]), torch.as_tensor([[0.2, 0.3]]), torch.as_tensor([[0.3, 0.4]])],
            [torch.as_tensor(0.1), torch.as_tensor(0.2), torch.as_tensor(0.3)],
        ]
        expected = [torch.as_tensor([0.2]), torch.as_tensor([0.15]), torch.as_tensor([0.2, 0.3]), torch.as_tensor(0.2)]
        average = CumulativeAverage()

        for i, e in zip(input_data, expected):
            func = average.append if i[0].ndim < 2 else average.extend
            if rank == 0:
                func(i[0])
                func(i[1])
            else:
                func(i[2])
            result = average.aggregate()
            assert_allclose(result, e)
            average.reset()


if __name__ == "__main__":
    unittest.main()
