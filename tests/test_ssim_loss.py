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

from monai.losses.ssim_loss import SSIMLoss

x = torch.ones([1, 1, 10, 10]) / 2
y1 = torch.ones([1, 1, 10, 10]) / 2
y2 = torch.zeros([1, 1, 10, 10])
data_range = x.max().unsqueeze(0)
TESTS2D = []
for device in [None, "cpu", "cuda"] if torch.cuda.is_available() else [None, "cpu"]:
    TESTS2D.append((x.to(device), y1.to(device), data_range.to(device), torch.tensor(1.0).unsqueeze(0).to(device)))
    TESTS2D.append((x.to(device), y2.to(device), data_range.to(device), torch.tensor(0.0).unsqueeze(0).to(device)))

x = torch.ones([1, 1, 10, 10, 10]) / 2
y1 = torch.ones([1, 1, 10, 10, 10]) / 2
y2 = torch.zeros([1, 1, 10, 10, 10])
data_range = x.max().unsqueeze(0)
TESTS3D = []
for device in [None, "cpu", "cuda"] if torch.cuda.is_available() else [None, "cpu"]:
    TESTS3D.append((x.to(device), y1.to(device), data_range.to(device), torch.tensor(1.0).unsqueeze(0).to(device)))
    TESTS3D.append((x.to(device), y2.to(device), data_range.to(device), torch.tensor(0.0).unsqueeze(0).to(device)))


class TestSSIMLoss(unittest.TestCase):
    @parameterized.expand(TESTS2D)
    def test2d(self, x, y, drange, res):
        result = 1 - SSIMLoss(spatial_dims=2)(x, y, drange)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertTrue(torch.abs(res - result).item() < 0.001)

    @parameterized.expand(TESTS3D)
    def test3d(self, x, y, drange, res):
        result = 1 - SSIMLoss(spatial_dims=3)(x, y, drange)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertTrue(torch.abs(res - result).item() < 0.001)


if __name__ == "__main__":
    unittest.main()
