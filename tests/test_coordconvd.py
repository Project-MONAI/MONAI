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

from monai.transforms import CoordConvd

TEST_CASE_1 = [
    {"spatial_channels": (1, 2, 3), "keys": ["img"]},
    {"img": np.random.randint(0, 2, size=(1, 3, 3, 3))},
    (4, 3, 3, 3),
]

TEST_CASE_2 = [
    {"spatial_channels": (1,), "keys": ["img"]},
    {"img": np.random.randint(0, 2, size=(1, 3, 3, 3))},
    (2, 3, 3, 3),
]

TEST_CASE_ERROR_3 = [{"spatial_channels": (3,), "keys": ["img"]}, {"img": np.random.randint(0, 2, size=(1, 3, 3))}]

TEST_CASE_ERROR_4 = [{"spatial_channels": (0, 1, 2), "keys": ["img"]}, {"img": np.random.randint(0, 2, size=(1, 3, 3))}]


class TestCoordConv(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_shape(self, input_param, input, expected_shape):
        result = CoordConvd(**input_param)(input)
        self.assertEqual(list(result["img"].shape), list(expected_shape))

    @parameterized.expand([TEST_CASE_ERROR_3])
    def test_max_channel(self, input_param, input):
        with self.assertRaises(ValueError):
            CoordConvd(**input_param)(input)

    @parameterized.expand([TEST_CASE_ERROR_4])
    def test_channel_dim(self, input_param, input):
        with self.assertRaises(ValueError):
            CoordConvd(**input_param)(input)


if __name__ == "__main__":
    unittest.main()
