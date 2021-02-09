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

from monai.transforms import CropForegroundd

TEST_CASE_1 = [
    {
        "keys": ["img", "label"],
        "source_key": "label",
        "select_fn": lambda x: x > 0,
        "channel_indices": None,
        "margin": 0,
    },
    {
        "img": np.array([[[1, 0, 2, 0, 1], [0, 1, 2, 1, 0], [2, 2, 3, 2, 2], [0, 1, 2, 1, 0], [1, 0, 2, 0, 1]]]),
        "label": np.array([[[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0]]]),
    },
    np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]]]),
]

TEST_CASE_2 = [
    {"keys": ["img"], "source_key": "img", "select_fn": lambda x: x > 1, "channel_indices": None, "margin": 0},
    {"img": np.array([[[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 3, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]]])},
    np.array([[[3]]]),
]

TEST_CASE_3 = [
    {"keys": ["img"], "source_key": "img", "select_fn": lambda x: x > 0, "channel_indices": 0, "margin": 0},
    {"img": np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]])},
    np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]]]),
]

TEST_CASE_4 = [
    {"keys": ["img"], "source_key": "img", "select_fn": lambda x: x > 0, "channel_indices": None, "margin": 1},
    {"img": np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]])},
    np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 0, 0, 0, 0]]]),
]

TEST_CASE_5 = [
    {"keys": ["img"], "source_key": "img", "select_fn": lambda x: x > 0, "channel_indices": None, "margin": [2, 1]},
    {"img": np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]])},
    np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]),
]


class TestCropForegroundd(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5])
    def test_value(self, arguments, image, expected_data):
        result = CropForegroundd(**arguments)(image)
        np.testing.assert_allclose(result["img"], expected_data)

    @parameterized.expand([TEST_CASE_1])
    def test_foreground_position(self, arguments, image, _):
        result = CropForegroundd(**arguments)(image)
        np.testing.assert_allclose(result["foreground_start_coord"], np.array([1, 1]))
        np.testing.assert_allclose(result["foreground_end_coord"], np.array([4, 4]))

        arguments["start_coord_key"] = "test_start_coord"
        arguments["end_coord_key"] = "test_end_coord"
        result = CropForegroundd(**arguments)(image)
        np.testing.assert_allclose(result["test_start_coord"], np.array([1, 1]))
        np.testing.assert_allclose(result["test_end_coord"], np.array([4, 4]))


if __name__ == "__main__":
    unittest.main()
