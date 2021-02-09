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

from monai.apps.deepgrow.transforms import FindAllValidSlicesd

TEST_CASE_1 = [
    {
        "image": np.array([[[[1, 0, 2, 0, 1], [0, 1, 2, 1, 0], [2, 2, 3, 2, 2], [0, 1, 2, 1, 0], [1, 0, 2, 0, 1]]]]),
        "label": np.array(
            [[np.zeros((5, 5)), [[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0]]]]
        ),
    },
    [1],
]

TEST_CASE_2 = [
    {
        "image": np.array([[[[1, 0, 2, 0, 1], [0, 1, 2, 1, 0], [2, 2, 3, 2, 2], [0, 1, 2, 1, 0], [1, 0, 2, 0, 1]]]]),
        "label": np.zeros((1, 3, 5, 5)),
    },
    None,
]


class TestFindAllValidSlicesd(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_correct_results(self, input_data, expected_result):
        result = FindAllValidSlicesd()(input_data)
        assert result.get("sids", None) == expected_result


if __name__ == "__main__":
    unittest.main()
