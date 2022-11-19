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

from monai.metrics import CumulativeAverage
from tests.utils import DistTestCase, SkipIfBeforePyTorchVersion


@SkipIfBeforePyTorchVersion((1, 8))
class DistributedCumulativeAverage(DistTestCase):
    def test_value(self):

        rank = 0
        is_cuda = torch.cuda.is_available()
        device = torch.device(rank) if is_cuda else torch.device("cpu")

        print('simple val')
        val = torch.tensor(0).to(device=device)

        print('simple val2')
        val = torch.as_tensor(0, device=device)

        avg_meter = CumulativeAverage()  # each process rank has it's own AverageMeter
        n_iter = 10
        for i in range(n_iter):
            val = torch.as_tensor(1, device=device)
            avg_meter.append(val=val)

        avg_val = avg_meter.aggregate()  # average across all processes
        expected_val = 1
        np.testing.assert_equal(avg_val, expected_val)


if __name__ == "__main__":
    unittest.main()
