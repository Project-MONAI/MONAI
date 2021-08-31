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

from monai.transforms import LocalPatchShuffling

TEST_CASES = [
    [
        {"number_blocks": 10, "blocksize_ratio": 1, "prob": 0.0},
        {"img": np.arange(8).reshape((1, 2, 2, 2))},
        np.arange(8).reshape((1, 2, 2, 2)),
    ],
    [
        {"number_blocks": 10, "blocksize_ratio": 1, "prob": 1.0},
        {"img": np.arange(27).reshape((1, 3, 3, 3))},
        [
            [
                [[9, 1, 2], [3, 4, 5], [6, 7, 8]],
                [[0, 10, 11], [12, 4, 14], [15, 16, 17]],
                [[18, 19, 20], [21, 22, 23], [24, 25, 26]],
            ]
        ],
    ],
]


class TestLocalPatchShuffle(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_local_patch_shuffle(self, input_param, input_data, expected_val):
        g = LocalPatchShuffling(**input_param)
        g.set_random_state(seed=12)
        result = g(**input_data)
        np.testing.assert_allclose(result, expected_val, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
