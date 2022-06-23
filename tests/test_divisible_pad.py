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

from monai.transforms import DivisiblePad
from tests.utils import TEST_NDARRAYS_ALL

TESTS = []

for p in TEST_NDARRAYS_ALL:
    # pad first dim to be divisible by 7, the second unchanged.
    TESTS.append([{"k": (7, -1), "mode": "constant"}, p(np.zeros((3, 8, 7))), p(np.zeros((3, 14, 7)))])

    # pad all dimensions to be divisible by 5
    TESTS.append(
        [{"k": 5, "mode": "constant", "method": "end"}, p(np.zeros((3, 10, 5, 17))), p(np.zeros((3, 10, 5, 20)))]
    )


class TestDivisiblePad(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_pad_shape(self, input_param, input_data, expected_val):
        padder = DivisiblePad(**input_param)
        result = padder(input_data)
        self.assertAlmostEqual(result.shape, expected_val.shape)
        result = padder(input_data, mode=input_param["mode"])
        self.assertAlmostEqual(result.shape, expected_val.shape)

    def test_pad_kwargs(self):
        for p in TEST_NDARRAYS_ALL:
            input_data = p(np.zeros((3, 8, 4)))
            if isinstance(input_data, np.ndarray):
                result = DivisiblePad(k=5, mode="constant", constant_values=((0, 0), (1, 1), (2, 2)))(input_data)
                np.testing.assert_allclose(result[:, :1, :4], np.ones((3, 1, 4)), rtol=1e-7, atol=0)
            else:
                result = DivisiblePad(k=5, mode="constant", value=2)(input_data).cpu()
            torch.testing.assert_allclose(result[:, :, 4:5], np.ones((3, 10, 1)) + 1, rtol=1e-7, atol=0)


if __name__ == "__main__":
    unittest.main()
