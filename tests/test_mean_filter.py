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

from monai.networks.layers import MeanFilter

TEST_CASES = [
    {"spatial_dims": 3, "size": 3, "expected": torch.ones(3, 3, 3)},
    {"spatial_dims": 2, "size": 5, "expected": torch.ones(5, 5)},
]


class MedianFilterTestCase(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_init(self, spatial_dims, size, expected):
        mean_filter = MeanFilter(spatial_dims=spatial_dims, size=size)
        self.assertEqual(expected, mean_filter.filter)
        self.assertIsInstance(mean_filter, torch.nn.Module)

    def test_forward(self):
        mean_filter = MeanFilter(spatial_dims=2, size=3)
        input = torch.ones(1, 1, 5, 5)
        output = mean_filter(input)
        self.assertEqual(input, output)


if __name__ == "__main__":
    unittest.main()
