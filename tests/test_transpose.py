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

from monai.transforms import Transpose

TEST_CASE_0 = [
    np.arange(5 * 4).reshape(5, 4),
    None,
]
TEST_CASE_1 = [
    np.arange(5 * 4 * 3).reshape(5, 4, 3),
    [2, 0, 1],
]
TEST_CASES = [TEST_CASE_0, TEST_CASE_1]


class TestTranspose(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_transpose(self, im, indices):
        tr = Transpose(indices)
        for array_class in (np.array, torch.Tensor):
            out1 = tr(array_class(im))
            out2 = np.transpose(im, indices)
            np.testing.assert_array_equal(out1, out2)


if __name__ == "__main__":
    unittest.main()
