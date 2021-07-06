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
import torch

from monai.transforms import MaskIntensity
from tests.utils import TEST_NDARRAYS

TEST_CASES = []

for p in TEST_NDARRAYS:
    for q in TEST_NDARRAYS:

        TEST_CASES.append(
            [
                {"mask_data": p([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])},
                q([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]]),
                q([[[0, 0, 0], [0, 2, 0], [0, 0, 0]], [[0, 0, 0], [0, 5, 0], [0, 0, 0]]]),
            ]
        )
        TEST_CASES.append(
            [
                {"mask_data": p([[[0, 0, 0], [0, 5, 0], [0, 0, 0]]])},
                q([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]]),
                q([[[0, 0, 0], [0, 2, 0], [0, 0, 0]], [[0, 0, 0], [0, 5, 0], [0, 0, 0]]]),
            ]
        )
        TEST_CASES.append(
            [
                {"mask_data": p([[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [0, 1, 0], [0, 1, 0]]])},
                q([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]]),
                q([[[0, 0, 0], [0, 2, 0], [0, 0, 0]], [[0, 4, 0], [0, 5, 0], [0, 6, 0]]]),
            ]
        )


class TestMaskIntensity(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_value(self, argments, image, expected_data):
        result = MaskIntensity(**argments)(image)
        torch.testing.assert_allclose(result, expected_data, rtol=1e-7, atol=0)


if __name__ == "__main__":
    unittest.main()
