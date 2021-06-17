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
from copy import deepcopy

import numpy as np
from parameterized import parameterized

from monai.transforms import Transposed

TEST_CASE_0 = [
    np.arange(5 * 4).reshape(5, 4),
    [1, 0],
]
TEST_CASE_1 = [
    np.arange(5 * 4).reshape(5, 4),
    None,
]
TEST_CASE_2 = [
    np.arange(5 * 4 * 3).reshape(5, 4, 3),
    [2, 0, 1],
]
TEST_CASE_3 = [
    np.arange(5 * 4 * 3).reshape(5, 4, 3),
    None,
]
TEST_CASES = [TEST_CASE_0, TEST_CASE_1, TEST_CASE_2, TEST_CASE_3]


class TestTranspose(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_transpose(self, im, indices):
        data = {"i": deepcopy(im), "j": deepcopy(im)}
        tr = Transposed(["i", "j"], indices)
        out_data = tr(data)
        out_im1, out_im2 = out_data["i"], out_data["j"]
        out_gt = np.transpose(im, indices)
        np.testing.assert_array_equal(out_im1, out_gt)
        np.testing.assert_array_equal(out_im2, out_gt)

        # test inverse
        fwd_inv_data = tr.inverse(out_data)
        for i, j in zip(data.values(), fwd_inv_data.values()):
            np.testing.assert_array_equal(i, j)


if __name__ == "__main__":
    unittest.main()
