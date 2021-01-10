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

from monai.transforms import SpatialPad

TEST_CASE_1 = [
    {"spatial_size": [15, 8, 8], "method": "symmetric", "mode": "constant"},
    np.zeros((3, 8, 8, 4)),
    np.zeros((3, 15, 8, 8)),
]

TEST_CASE_2 = [
    {"spatial_size": [15, 8, 8], "method": "end", "mode": "constant"},
    np.zeros((3, 8, 8, 4)),
    np.zeros((3, 15, 8, 8)),
]

TEST_CASE_3 = [
    {"spatial_size": [15, 4, -1], "method": "symmetric", "mode": "constant"},
    np.zeros((3, 8, 8, 4)),
    np.zeros((3, 15, 8, 4)),
]


class TestSpatialPad(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_pad_shape(self, input_param, input_data, expected_val):
        padder = SpatialPad(**input_param)
        result = padder(input_data)
        np.testing.assert_allclose(result.shape, expected_val.shape)
        result = padder(input_data, mode=input_param["mode"])
        np.testing.assert_allclose(result.shape, expected_val.shape)


if __name__ == "__main__":
    unittest.main()
