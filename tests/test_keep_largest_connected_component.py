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

from monai.transforms import KeepLargestConnectedComponent

grid_1 = np.array([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [1, 2, 1, 0, 0], [1, 2, 0, 1, 0], [2, 2, 0, 0, 2]]])
grid_2 = np.array([[[0, 0, 0, 0, 1], [0, 0, 1, 1, 1], [1, 0, 1, 1, 2], [1, 0, 1, 2, 2], [0, 0, 0, 0, 1]]])

TEST_CASE_1 = [
    "independent_label_1",
    {"is_independent": True},
    grid_1,
    [1],
    np.array([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [0, 2, 1, 0, 0], [0, 2, 0, 1, 0], [2, 2, 0, 0, 2]]]),
]

TEST_CASE_2 = [
    "independent_label_2",
    {"is_independent": True},
    grid_1,
    [2],
    np.array([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [1, 2, 1, 0, 0], [1, 2, 0, 1, 0], [2, 2, 0, 0, 0]]]),
]

TEST_CASE_3 = [
    "independent_label_1_2",
    {"is_independent": True},
    grid_1,
    [1, 2],
    np.array([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [0, 2, 1, 0, 0], [0, 2, 0, 1, 0], [2, 2, 0, 0, 0]]]),
]

TEST_CASE_4 = [
    "non_independent_label_1_2",
    {"is_independent": False},
    grid_1,
    [1, 2],
    np.array([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [1, 2, 1, 0, 0], [1, 2, 0, 1, 0], [2, 2, 0, 0, 2]]]),
]

TEST_CASE_5 = [
    "independent_label_1",
    {"is_independent": True},
    grid_2,
    [1],
    np.array([[[0, 0, 0, 0, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 2], [0, 0, 1, 2, 2], [0, 0, 0, 0, 0]]]),
]

TEST_CASE_6 = [
    "independent_label_1_2",
    {"is_independent": True},
    grid_2,
    [1, 2],
    np.array([[[0, 0, 0, 0, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 2], [0, 0, 1, 2, 2], [0, 0, 0, 0, 0]]]),
]

TEST_CASE_7 = [
    "non_independent_label_1_2",
    {"is_independent": False},
    grid_2,
    [1, 2],
    np.array([[[0, 0, 0, 0, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 2], [0, 0, 1, 2, 2], [0, 0, 0, 0, 1]]]),
]

VALID_CASES = [TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4,
               TEST_CASE_5, TEST_CASE_6, TEST_CASE_7]


class TestKeepLargestConnectedComponent(unittest.TestCase):
    @parameterized.expand(VALID_CASES)
    def test_correct_results(self, _, args, array, applied_labels, expected):
        converter = KeepLargestConnectedComponent(**args)
        result = converter(array, applied_labels)
        np.testing.assert_allclose(result, expected)


if __name__ == "__main__":
    unittest.main()
