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

from monai.transforms import Transpose

TEST_CASES = []
for q in (np.arange, torch.Tensor):
    TEST_CASES.append([q(5 * 4).reshape(5, 4), None])
    TEST_CASES.append([q(5 * 4 * 3).reshape(5, 4, 3), [2, 0, 1]])


class TestTranspose(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_transpose(self, im, indices):
        tr = Transpose(indices)

        out1 = tr(im)
        out2 = np.transpose(im, indices)
        np.testing.assert_array_equal(out1, out2)


if __name__ == "__main__":
    unittest.main()
