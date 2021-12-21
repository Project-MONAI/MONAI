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
from parameterized import parameterized

from monai.optimizers.lr_scheduler import WarmupCosineSchedule


class SchedulerTestNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv2(torch.nn.functional.relu(self.conv1(x)))


TEST_CASE_LRSCHEDULER = [
    [{"warmup_steps": 2, "t_total": 10}, [0.000, 0.500, 1.00, 0.962, 0.854, 0.691, 0.500, 0.309, 0.146, 0.038]]
]


class TestLRSCHEDULER(unittest.TestCase):
    @parameterized.expand(TEST_CASE_LRSCHEDULER)
    def test_shape(self, input_param, expected_lr):
        net = SchedulerTestNet()
        optimizer = torch.optim.Adam(net.parameters(), lr=1.0)
        scheduler = WarmupCosineSchedule(optimizer, **input_param)
        self.assertEqual(len([scheduler.get_last_lr()[0]]), 1)
        lrs_1 = []
        for _ in range(input_param["t_total"]):
            lrs_1.append(float(f"{scheduler.get_last_lr()[0]:.3f}"))
            optimizer.step()
            scheduler.step()
        for a, b in zip(lrs_1, expected_lr):
            self.assertEqual(a, b, msg=f"LR is wrong ! expected {b}, got {a}")


if __name__ == "__main__":
    unittest.main()
