# Copyright 2020 MONAI Consortium
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

from monai.transforms import GaussianSmooth

TEST_CASE_1 = [
    {"sigma": 1.5},
    np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]]),
    np.array(
        [
            [[0.5999930, 0.7056839, 0.5999930], [0.8140513, 0.9574494, 0.8140513], [0.7842673, 0.9224188, 0.7842673]],
            [[1.6381884, 1.926761, 1.6381884], [2.0351284, 2.3936234, 2.0351284], [1.8224627, 2.143496, 1.8224627]],
        ]
    ),
]

TEST_CASE_2 = [
    {"sigma": 0.5},
    np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]]),
    np.array(
        [
            [[0.893521, 0.99973595, 0.893521], [1.785628, 1.9978896, 1.7856278], [2.2983139, 2.5715199, 2.2983139]],
            [[3.2873974, 3.6781778, 3.2873974], [4.46407, 4.9947243, 4.46407], [4.69219, 5.2499614, 4.69219]],
        ]
    ),
]

TEST_CASE_3 = [
    {"sigma": [1.5, 0.5]},
    np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]]),
    np.array(
        [
            [[0.91108215, 1.0193846, 0.91108215], [1.236127, 1.3830683, 1.236127], [1.1909003, 1.3324654, 1.1909003]],
            [[2.4875693, 2.7832723, 2.487569], [3.0903177, 3.457671, 3.0903175], [2.7673876, 3.0963533, 2.7673874]],
        ]
    ),
]


class TestGaussianSmooth(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_value(self, argments, image, expected_data):
        result = GaussianSmooth(**argments)(image)
        np.testing.assert_allclose(result, expected_data, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
