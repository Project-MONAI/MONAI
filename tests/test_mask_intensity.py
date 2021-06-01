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
import torch
from parameterized import parameterized

from monai.transforms import MaskIntensity

TEST_CASES = []
for p in (torch.Tensor, np.array):
    for q in (torch.Tensor, np.array):
        for r in (torch.Tensor, np.array):
            TEST_CASES.append(
                [
                    {"mask_data": p([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])},  # type: ignore
                    q([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]]),  # type: ignore
                    r([[[0, 0, 0], [0, 2, 0], [0, 0, 0]], [[0, 0, 0], [0, 5, 0], [0, 0, 0]]]),  # type: ignore
                ]
            )
            TEST_CASES.append(
                [
                    {"mask_data": p([[[0, 0, 0], [0, 5, 0], [0, 0, 0]]])},  # type: ignore
                    q([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]]),  # type: ignore
                    r([[[0, 0, 0], [0, 2, 0], [0, 0, 0]], [[0, 0, 0], [0, 5, 0], [0, 0, 0]]]),  # type: ignore
                ]
            )
            TEST_CASES.append(
                [
                    {"mask_data": p([[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [0, 1, 0], [0, 1, 0]]])},  # type: ignore
                    q([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]]),  # type: ignore
                    r([[[0, 0, 0], [0, 2, 0], [0, 0, 0]], [[0, 4, 0], [0, 5, 0], [0, 6, 0]]]),  # type: ignore
                ]
            )


class TestMaskIntensity(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_value(self, argments, image, expected_data):
        result = MaskIntensity(**argments)(image)
        np.testing.assert_allclose(result, expected_data)


if __name__ == "__main__":
    unittest.main()
