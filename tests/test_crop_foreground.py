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
from functools import partial
from typing import Callable, List

import numpy as np
import torch
from parameterized import parameterized

from monai.transforms import CropForeground

TESTS = []
NDARRAYS: List[Callable] = [np.array, torch.Tensor]
if torch.cuda.is_available():
    NDARRAYS.append(partial(torch.Tensor, device="cuda"))

for p in NDARRAYS:
    TESTS.append(
        [
            {"select_fn": lambda x: x > 0, "channel_indices": None, "margin": 0},
            p([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]]),  # type: ignore
            np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]]]),
        ]
    )

    TESTS.append(
        [
            {"select_fn": lambda x: x > 1, "channel_indices": None, "margin": 0},
            p([[[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 3, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]]]),  # type: ignore
            np.array([[[3]]]),
        ]
    )

    TESTS.append(
        [
            {"select_fn": lambda x: x > 0, "channel_indices": 0, "margin": 0},
            p([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]]),  # type: ignore
            np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]]]),
        ]
    )

    TESTS.append(
        [
            {"select_fn": lambda x: x > 0, "channel_indices": None, "margin": 1},
            p([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]),  # type: ignore
            np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 0, 0, 0, 0]]]),
        ]
    )

    TESTS.append(
        [
            {"select_fn": lambda x: x > 0, "channel_indices": None, "margin": [2, 1]},
            p([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]),  # type: ignore
            np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]),
        ]
    )

    TESTS.append(
        [
            {"select_fn": lambda x: x > 0, "channel_indices": None, "margin": 0, "k_divisible": 4},
            p([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]]),  # type: ignore
            np.array([[[1, 2, 1, 0], [2, 3, 2, 0], [1, 2, 1, 0], [0, 0, 0, 0]]]),
        ]
    )

    TESTS.append(
        [
            {"select_fn": lambda x: x > 0, "channel_indices": None, "margin": 0, "k_divisible": 10},
            p([[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]),  # type: ignore
            np.zeros((1, 0, 0)),
        ]
    )


class TestCropForeground(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_value(self, argments, image, expected_data):
        result = CropForeground(**argments)(image)
        np.testing.assert_allclose(result, expected_data)

    @parameterized.expand([TESTS[0]])
    def test_return_coords(self, argments, image, _):
        argments["return_coords"] = True
        _, start_coord, end_coord = CropForeground(**argments)(image)
        argments["return_coords"] = False
        np.testing.assert_allclose(start_coord, np.asarray([1, 1]))
        np.testing.assert_allclose(end_coord, np.asarray([4, 4]))


if __name__ == "__main__":
    unittest.main()
