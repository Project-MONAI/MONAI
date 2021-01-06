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

from monai.transforms import ThresholdIntensity

TEST_CASE_1 = [{"threshold": 5, "above": True, "cval": 0}, (0, 0, 0, 0, 0, 0, 6, 7, 8, 9)]

TEST_CASE_2 = [{"threshold": 5, "above": False, "cval": 0}, (0, 1, 2, 3, 4, 0, 0, 0, 0, 0)]

TEST_CASE_3 = [{"threshold": 5, "above": True, "cval": 5}, (5, 5, 5, 5, 5, 5, 6, 7, 8, 9)]


class TestThresholdIntensity(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_value(self, input_param, expected_value):
        test_data = np.arange(10)
        result = ThresholdIntensity(**input_param)(test_data)
        np.testing.assert_allclose(result, expected_value)


if __name__ == "__main__":
    unittest.main()
