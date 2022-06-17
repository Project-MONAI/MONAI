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

from monai.transforms import CropForegroundd
from tests.utils import TEST_NDARRAYS, assert_allclose

TEST_POSITION, TESTS = [], []
for p in TEST_NDARRAYS:

    TEST_POSITION.append(
        [
            {
                "keys": ["img", "label"],
                "source_key": "label",
                "select_fn": lambda x: x > 0,
                "channel_indices": None,
                "margin": 0,
            },
            {
                "img": p(
                    np.array([[[1, 0, 2, 0, 1], [0, 1, 2, 1, 0], [2, 2, 3, 2, 2], [0, 1, 2, 1, 0], [1, 0, 2, 0, 1]]])
                ),
                "label": p(
                    np.array([[[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0]]])
                ),
            },
            p(np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]]])),
        ]
    )
    TESTS.append(
        [
            {"keys": ["img"], "source_key": "img", "select_fn": lambda x: x > 1, "channel_indices": None, "margin": 0},
            {
                "img": p(
                    np.array([[[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 3, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]]])
                )
            },
            p(np.array([[[3]]])),
        ]
    )
    TESTS.append(
        [
            {"keys": ["img"], "source_key": "img", "select_fn": lambda x: x > 0, "channel_indices": 0, "margin": 0},
            {
                "img": p(
                    np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]])
                )
            },
            p(np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]]])),
        ]
    )
    TESTS.append(
        [
            {"keys": ["img"], "source_key": "img", "select_fn": lambda x: x > 0, "channel_indices": None, "margin": 1},
            {
                "img": p(
                    np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]])
                )
            },
            p(np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 0, 0, 0, 0]]])),
        ]
    )
    TESTS.append(
        [
            {
                "keys": ["img"],
                "source_key": "img",
                "select_fn": lambda x: x > 0,
                "channel_indices": None,
                "margin": [2, 1],
                "allow_smaller": True,
            },
            {
                "img": p(
                    np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]])
                )
            },
            p(np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]])),
        ]
    )
    TESTS.append(
        [
            {
                "keys": ["img"],
                "source_key": "img",
                "select_fn": lambda x: x > 0,
                "channel_indices": None,
                "margin": [2, 1],
                "allow_smaller": False,
            },
            {
                "img": p(
                    np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]])
                )
            },
            p(
                np.array(
                    [
                        [
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 2, 1, 0],
                            [0, 2, 3, 2, 0],
                            [0, 1, 2, 1, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ]
                )
            ),
        ]
    )
    TESTS.append(
        [
            {
                "keys": ["img"],
                "source_key": "img",
                "select_fn": lambda x: x > 0,
                "channel_indices": 0,
                "margin": 0,
                "k_divisible": [4, 6],
                "mode": "edge",
            },
            {
                "img": p(
                    np.array(
                        [[[0, 2, 1, 2, 0], [1, 1, 2, 1, 1], [2, 2, 3, 2, 2], [1, 1, 2, 1, 1], [0, 0, 0, 0, 0]]],
                        dtype=np.float32,
                    )
                )
            },
            p(np.array([[[0, 2, 1, 2, 0, 0], [1, 1, 2, 1, 1, 1], [2, 2, 3, 2, 2, 2], [1, 1, 2, 1, 1, 1]]])),
        ]
    )


class TestCropForegroundd(unittest.TestCase):
    @parameterized.expand(TEST_POSITION + TESTS)
    def test_value(self, argments, input_data, expected_data):
        result = CropForegroundd(**argments)(input_data)
        r, i = result["img"], input_data["img"]
        self.assertEqual(type(r), type(i))
        if isinstance(r, torch.Tensor):
            self.assertEqual(r.device, i.device)
        assert_allclose(r, expected_data)

    @parameterized.expand(TEST_POSITION)
    def test_foreground_position(self, argments, input_data, _):
        result = CropForegroundd(**argments)(input_data)
        np.testing.assert_allclose(result["foreground_start_coord"], np.array([1, 1]))
        np.testing.assert_allclose(result["foreground_end_coord"], np.array([4, 4]))

        argments["start_coord_key"] = "test_start_coord"
        argments["end_coord_key"] = "test_end_coord"
        result = CropForegroundd(**argments)(input_data)
        np.testing.assert_allclose(result["test_start_coord"], np.array([1, 1]))
        np.testing.assert_allclose(result["test_end_coord"], np.array([4, 4]))


if __name__ == "__main__":
    unittest.main()
