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

from monai.metrics.regression import SSIMMetric

x = torch.ones([1, 1, 10, 10]) / 2
y1 = torch.ones([1, 1, 10, 10]) / 2
y2 = torch.zeros([1, 1, 10, 10])
data_range = x.max().unsqueeze(0)
TESTS2D = [(x, y1, data_range, torch.tensor(1.0).unsqueeze(0)), (x, y2, data_range, torch.tensor(0.0).unsqueeze(0))]

x = torch.ones([1, 1, 10, 10, 10]) / 2
y1 = torch.ones([1, 1, 10, 10, 10]) / 2
y2 = torch.zeros([1, 1, 10, 10, 10])
data_range = x.max().unsqueeze(0)
TESTS3D = [(x, y1, data_range, torch.tensor(1.0).unsqueeze(0)), (x, y2, data_range, torch.tensor(0.0).unsqueeze(0))]

x = torch.ones([3, 3, 10, 10, 10]) / 2
y1 = torch.ones([3, 3, 10, 10, 10]) / 2
data_range = x.max().unsqueeze(0)
full = True
res = torch.tensor(1.0).unsqueeze(0) * torch.ones((3, 1))
TESTSFULL = [(x, y1, data_range, full, res)]


class TestSSIMMetric(unittest.TestCase):
    @parameterized.expand(TESTS2D)
    def test2d(self, x, y, drange, res):
        result = SSIMMetric(data_range=drange, spatial_dims=2)._compute_metric(x, y)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertTrue(torch.abs(res - result).item() < 0.001)

        ssim = SSIMMetric(data_range=drange, spatial_dims=2)
        ssim(x, y)
        result2 = ssim.aggregate()
        self.assertTrue(isinstance(result2, torch.Tensor))
        self.assertTrue(torch.abs(result2 - result).item() < 0.001)

    @parameterized.expand(TESTS3D)
    def test3d(self, x, y, drange, res):
        result = SSIMMetric(data_range=drange, spatial_dims=3)._compute_metric(x, y)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertTrue(torch.abs(res - result).item() < 0.001)

        ssim = SSIMMetric(data_range=drange, spatial_dims=3)
        ssim(x, y)
        result2 = ssim.aggregate()
        self.assertTrue(isinstance(result2, torch.Tensor))
        self.assertTrue(torch.abs(result2 - result).item() < 0.001)

    @parameterized.expand(TESTSFULL)
    def testfull(self, x, y, drange, full, res):
        result, cs = SSIMMetric(data_range=drange, spatial_dims=3)._compute_metric(x, y, full=full)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertTrue(isinstance(cs, torch.Tensor))
        self.assertTrue((torch.abs(res - cs) < 0.001).all().item())


if __name__ == "__main__":
    unittest.main()
