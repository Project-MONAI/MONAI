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
from parameterized import parameterized

from monai.transforms import DivisiblePadd

TEST_CASE_1 = [
    {"keys": ["img"], "k": [4, 3, 2], "mode": "constant"},
    {"img": np.zeros((3, 8, 8, 4))},
    np.zeros((3, 8, 9, 4)),
]

TEST_CASE_2 = [
    {"keys": ["img"], "k": 7, "mode": "constant", "method": "end"},
    {"img": np.zeros((3, 8, 7))},
    np.zeros((3, 14, 7)),
]

TEST_CASE_3 = [{"keys": ["img"], "k": 0, "mode": {"constant"}}, {"img": np.zeros((3, 8))}, np.zeros((3, 8))]


class TestDivisiblePadd(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_pad_shape(self, input_param, input_data, expected_val):
        padder = DivisiblePadd(**input_param)
        result = padder(input_data)
        np.testing.assert_allclose(result["img"], expected_val)


if __name__ == "__main__":
    unittest.main()
