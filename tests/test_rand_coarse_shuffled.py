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

from monai.transforms import RandCoarseShuffled

TEST_CASES = [
    [
        {"keys": "img", "holes": 5, "spatial_size": 1, "max_spatial_size": -1, "prob": 0.0},
        {"img": np.arange(8).reshape((1, 2, 2, 2))},
        np.arange(8).reshape((1, 2, 2, 2)),
    ],
    [
        {"keys": "img", "holes": 10, "spatial_size": 1, "max_spatial_size": -1, "prob": 1.0},
        {"img": np.arange(27).reshape((1, 3, 3, 3))},
        np.asarray(
            [
                [
                    [[13, 17, 5], [6, 16, 25], [12, 15, 22]],
                    [[24, 7, 3], [9, 2, 23], [0, 4, 26]],
                    [[19, 11, 14], [1, 20, 8], [18, 10, 21]],
                ]
            ]
        ),
    ],
    [
        {"keys": "img", "holes": 2, "spatial_size": 1, "max_spatial_size": -1, "prob": 1.0},
        {"img": np.arange(16).reshape((2, 2, 2, 2))},
        np.asarray([[[[7, 2], [1, 4]], [[5, 0], [3, 6]]], [[[8, 13], [10, 15]], [[14, 12], [11, 9]]]]),
    ],
]


class TestRandCoarseShuffled(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_shuffle(self, input_param, input_data, expected_val):
        g = RandCoarseShuffled(**input_param)
        g.set_random_state(seed=12)
        result = g(input_data)
        np.testing.assert_allclose(result["img"], expected_val, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
