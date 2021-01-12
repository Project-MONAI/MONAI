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

from monai.transforms import CropForeground

TEST_CASE_1 = [
    {"select_fn": lambda x: x > 0, "channel_indices": None, "margin": 0},
    np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]]),
    np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]]]),
]

TEST_CASE_2 = [
    {"select_fn": lambda x: x > 1, "channel_indices": None, "margin": 0},
    np.array([[[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 3, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]]]),
    np.array([[[3]]]),
]

TEST_CASE_3 = [
    {"select_fn": lambda x: x > 0, "channel_indices": 0, "margin": 0},
    np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]]),
    np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]]]),
]

TEST_CASE_4 = [
    {"select_fn": lambda x: x > 0, "channel_indices": None, "margin": 1},
    np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]),
    np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 0, 0, 0, 0]]]),
]

TEST_CASE_5 = [
    {"select_fn": lambda x: x > 0, "channel_indices": None, "margin": [2, 1]},
    np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]),
    np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]),
]


class TestCropForeground(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5])
    def test_value(self, argments, image, expected_data):
        result = CropForeground(**argments)(image)
        np.testing.assert_allclose(result, expected_data)

    @parameterized.expand([TEST_CASE_1])
    def test_return_coords(self, argments, image, _):
        argments["return_coords"] = True
        _, start_coord, end_coord = CropForeground(**argments)(image)
        argments["return_coords"] = False
        self.assertListEqual(start_coord, [1, 1])
        self.assertListEqual(end_coord, [4, 4])


if __name__ == "__main__":
    unittest.main()
