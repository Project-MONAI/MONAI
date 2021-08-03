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

from monai.transforms import IntensityStats

TEST_CASE_1 = [
    {"ops": ["max", "mean"], "key_prefix": "orig"},
    np.array([[[0.0, 1.0], [2.0, 3.0]]]),
    {"affine": None},
    {"orig_max": 3.0, "orig_mean": 1.5},
]

TEST_CASE_2 = [
    {"ops": "std", "key_prefix": "orig"},
    np.array([[[0.0, 1.0], [2.0, 3.0]]]),
    None,
    {"orig_std": 1.118034},
]

TEST_CASE_3 = [
    {"ops": [lambda x: np.mean(x), "max", lambda x: np.min(x)], "key_prefix": "orig"},
    np.array([[[0.0, 1.0], [2.0, 3.0]]]),
    None,
    {"orig_custom_0": 1.5, "orig_max": 3.0, "orig_custom_1": 0.0},
]

TEST_CASE_4 = [
    {"ops": ["max", "mean"], "key_prefix": "orig", "channel_wise": True},
    np.array([[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]]),
    {"affine": None},
    {"orig_max": [3.0, 7.0], "orig_mean": [1.5, 5.5]},
]


class TestIntensityStats(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4])
    def test_value(self, input_param, img, meta_dict, expected):
        img, meta_dict = IntensityStats(**input_param)(img, meta_dict)
        for k, v in expected.items():
            self.assertTrue(k in meta_dict)
            np.testing.assert_allclose(v, meta_dict[k], atol=1e-3)


if __name__ == "__main__":
    unittest.main()
