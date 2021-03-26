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

from monai.transforms import MapLabelValue

TEST_CASE_1 = [
    {"orig_labels": [3, 2, 1], "target_labels": [0, 1, 2]},
    np.array([[3, 1], [1, 2]]),
    np.array([[0, 2], [2, 1]]),
]

TEST_CASE_2 = [
    {"orig_labels": [3, 5, 8], "target_labels": [0, 1, 2]},
    np.array([[[3], [5], [5], [8]]]),
    np.array([[[0], [1], [1], [2]]]),
]

TEST_CASE_3 = [
    {"orig_labels": [1, 2, 3], "target_labels": [0, 1, 2]},
    np.array([3, 1, 1, 2]),
    np.array([2, 0, 0, 1]),
]


class TestMapLabelValue(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_shape(self, input_param, input_data, expected_value):
        result = MapLabelValue(**input_param)(input_data)
        np.testing.assert_allclose(result, expected_value)
        self.assertTupleEqual(result.shape, expected_value.shape)


if __name__ == "__main__":
    unittest.main()
