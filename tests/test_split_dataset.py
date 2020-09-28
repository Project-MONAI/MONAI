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

from parameterized import parameterized

from monai.apps import split_dataset

TEST_CASE_1 = [{"nsplits": 5, "length": 10}, [[0, 2], [2, 4], [4, 6], [6, 8], [8, 10]]]

TEST_CASE_2 = [{"nsplits": 5, "length": 13}, [[0, 3], [3, 6], [6, 9], [9, 11], [11, 13]]]

TEST_CASE_3 = [{"nsplits": 5, "length": 11}, [[0, 3], [3, 5], [5, 7], [7, 9], [9, 11]]]

TEST_CASE_4 = [{"nsplits": 1, "length": 10}, [[0, 10]]]


class TestSplitDataset(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4])
    def test_shape(self, input_param, expected_indices):
        indices = split_dataset(**input_param)
        for data, expected in zip(indices, expected_indices):
            self.assertListEqual(data, expected)


if __name__ == "__main__":
    unittest.main()
