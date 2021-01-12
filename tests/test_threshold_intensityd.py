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

from monai.transforms import ThresholdIntensityd

TEST_CASE_1 = [
    {"keys": ["image", "label", "extra"], "threshold": 5, "above": True, "cval": 0},
    (0, 0, 0, 0, 0, 0, 6, 7, 8, 9),
]

TEST_CASE_2 = [
    {"keys": ["image", "label", "extra"], "threshold": 5, "above": False, "cval": 0},
    (0, 1, 2, 3, 4, 0, 0, 0, 0, 0),
]

TEST_CASE_3 = [
    {"keys": ["image", "label", "extra"], "threshold": 5, "above": True, "cval": 5},
    (5, 5, 5, 5, 5, 5, 6, 7, 8, 9),
]


class TestThresholdIntensityd(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_value(self, input_param, expected_value):
        test_data = {"image": np.arange(10), "label": np.arange(10), "extra": np.arange(10)}
        result = ThresholdIntensityd(**input_param)(test_data)
        np.testing.assert_allclose(result["image"], expected_value)
        np.testing.assert_allclose(result["label"], expected_value)
        np.testing.assert_allclose(result["extra"], expected_value)


if __name__ == "__main__":
    unittest.main()
