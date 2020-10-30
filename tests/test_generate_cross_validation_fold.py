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

from monai.data import generate_cross_validation_fold, partition_dataset

TEST_CASE_1 = [
    {
        "data": [1, 2, 3, 4, 5],
        "num_partitions": 5,
        "shuffle": False,
        "seed": 0,
        "drop_last": False,
        "even_divisible": False,
    },
    [[1, 3, 4, 5], [2]],
]

TEST_CASE_2 = [
    {
        "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "num_partitions": 10,
        "shuffle": False,
        "seed": 0,
        "drop_last": False,
        "even_divisible": False,
    },
    [[1, 3, 4, 5, 6, 7, 8, 9, 10], [2]],
]


class TestGenerateCrossValidationFold(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_value(self, input_param, result):
        partitions = partition_dataset(**input_param)
        train, val = generate_cross_validation_fold(partitions, fold_idx=1)
        self.assertListEqual(train, result[0])
        self.assertListEqual(val, result[1])


if __name__ == "__main__":
    unittest.main()
