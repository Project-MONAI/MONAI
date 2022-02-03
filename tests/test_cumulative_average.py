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
from parameterized import parameterized

from monai.metrics import CumulativeAverage

# single class value
TEST_CASE_1 = [[torch.as_tensor([[0.1]]), torch.as_tensor([[0.2]]), torch.as_tensor([[0.3]])], torch.as_tensor([0.2])]

# multi-class value
TEST_CASE_2 = [
    [torch.as_tensor([[0.1, 0.2]]), torch.as_tensor([[0.2, 0.3]]), torch.as_tensor([[0.3, 0.4]])],
    torch.as_tensor([0.2, 0.3]),
]

# Nan value
TEST_CASE_3 = [
    [torch.as_tensor([[0.1]]), torch.as_tensor([[0.2]]), torch.as_tensor([[float("nan")]])],
    torch.as_tensor([0.15]),
]

# different input shape
TEST_CASE_4 = [[torch.as_tensor(0.1), torch.as_tensor(0.2), torch.as_tensor(0.3)], torch.as_tensor(0.2)]


class TestCumulativeAverage(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4])
    def test_value(self, input_data, expected_value):
        average = CumulativeAverage()
        func = average.append if input_data[0].ndim < 2 else average.extend
        func(input_data[0])
        func(input_data[1])
        result = average.aggregate()
        # continue to update new data
        func(input_data[2])
        result = average.aggregate()
        torch.testing.assert_allclose(result, expected_value)

    def test_numpy_array(self):
        class TestCumulativeAverage(CumulativeAverage):
            def get_buffer(self):
                return np.array([[1, 2], [3, np.nan]])

        average = TestCumulativeAverage()
        result = average.aggregate()
        np.testing.assert_allclose(result, np.array([2.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
