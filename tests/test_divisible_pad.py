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

import numpy as np
from parameterized import parameterized

from monai.transforms import DivisiblePad

# pad first dim to be divisible by 7, the second unchanged.
TEST_CASE_1 = [
    {"k": (7, -1), "mode": "constant"},
    np.zeros((3, 8, 7)),
    np.zeros((3, 14, 7)),
]

# pad all dimensions to be divisible by 5
TEST_CASE_2 = [
    {"k": 5, "mode": "constant", "method": "end"},
    np.zeros((3, 10, 5, 17)),
    np.zeros((3, 10, 5, 20)),
]


class TestDivisiblePad(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_pad_shape(self, input_param, input_data, expected_val):
        padder = DivisiblePad(**input_param)
        result = padder(input_data)
        self.assertAlmostEqual(result.shape, expected_val.shape)
        result = padder(input_data, mode=input_param["mode"])
        self.assertAlmostEqual(result.shape, expected_val.shape)

    def test_pad_kwargs(self):
        padder = DivisiblePad(k=5, mode="constant", constant_values=((0, 0), (1, 1), (2, 2)))
        result = padder(np.zeros((3, 8, 4)))
        np.testing.assert_allclose(result[:, :1, :4], np.ones((3, 1, 4)))
        np.testing.assert_allclose(result[:, :, 4:5], np.ones((3, 10, 1)) + 1)


if __name__ == "__main__":
    unittest.main()
