# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import unittest

from parameterized import parameterized

from monai.data import partition_dataset

TEST_CASE_1 = [
    {
        "data": [1, 2, 3, 4],
        "num_partitions": 2,
        "shuffle": False,
        "seed": 0,
        "drop_last": False,
        "even_divisible": False,
    },
    [[1, 3], [2, 4]],
]

TEST_CASE_2 = [
    {
        "data": [1, 2, 3, 4],
        "num_partitions": 2,
        "shuffle": True,
        "seed": 123,
        "drop_last": False,
        "even_divisible": False,
    },
    [[4, 2], [1, 3]],
]

TEST_CASE_3 = [
    {
        "data": [1, 2, 3, 4, 5],
        "num_partitions": 2,
        "shuffle": False,
        "seed": 0,
        "drop_last": False,
        "even_divisible": False,
    },
    [[1, 3, 5], [2, 4]],
]

TEST_CASE_4 = [
    {
        "data": [1, 2, 3, 4, 5],
        "num_partitions": 2,
        "shuffle": False,
        "seed": 0,
        "drop_last": False,
        "even_divisible": True,
    },
    [[1, 3, 5], [2, 4, 1]],
]

TEST_CASE_5 = [
    {
        "data": [1, 2, 3, 4, 5],
        "num_partitions": 2,
        "shuffle": False,
        "seed": 0,
        "drop_last": True,
        "even_divisible": True,
    },
    [[1, 3], [2, 4]],
]

TEST_CASE_6 = [
    {
        "data": [1, 2, 3, 4, 5],
        "ratios": [3, 2],
        "num_partitions": None,
        "shuffle": False,
        "seed": 0,
        "drop_last": True,
        "even_divisible": True,
    },
    [[1, 2, 3], [4, 5]],
]

TEST_CASE_7 = [
    {
        "data": [1, 2, 3, 4, 5],
        "ratios": [2, 1],
        "num_partitions": None,
        "shuffle": False,
        "seed": 0,
        "drop_last": True,
        "even_divisible": True,
    },
    [[1, 2, 3], [4, 5]],
]

TEST_CASE_8 = [
    {
        "data": [1, 2, 3, 4, 5],
        "ratios": [2, 1],
        "num_partitions": None,
        "shuffle": True,
        "seed": 123,
        "drop_last": True,
        "even_divisible": True,
    },
    [[2, 4, 5], [1, 3]],
]

# rounding each partition size on its own leaves one item over in these cases
TEST_CASES_ROUNDING = [
    [{"data": list(range(1, 11)), "ratios": [1, 1, 1]}, [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10]]],
    [{"data": list(range(1, 14)), "ratios": [0.8, 0.1, 0.1]}, [list(range(1, 12)), [12], [13]]],
]


TEST_CASE_9 = [
    {
        "data": list(range(10)),
        "ratios": [1, 1, 1],
        "num_partitions": None,
        "shuffle": False,
        "seed": 0,
        "drop_last": True,
        "even_divisible": True,
    },
    [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]],
]

TEST_CASE_10 = [
    {
        "data": list(range(63)),
        "ratios": [0.8, 0.1, 0.1],
        "num_partitions": None,
        "shuffle": False,
        "seed": 0,
        "drop_last": True,
        "even_divisible": True,
    },
    [list(range(51)), list(range(51, 57)), list(range(57, 63))],
]

TEST_CASE_11 = [
    {
        "data": list(range(32)),
        "ratios": [1, 1, 1, 1, 1],
        "num_partitions": None,
        "shuffle": False,
        "seed": 0,
        "drop_last": True,
        "even_divisible": True,
    },
    [list(range(7)), list(range(7, 14)), list(range(14, 20)), list(range(20, 26)), list(range(26, 32))],
]

TEST_CASE_12 = [
    {
        "data": [1, 2, 3, 4, 5],
        "ratios": [0.6, 0.2, 0.2],
        "num_partitions": None,
        "shuffle": False,
        "seed": 0,
        "drop_last": True,
        "even_divisible": True,
    },
    [[1, 2, 3], [4], [5]],
]

TEST_CASE_13 = [
    {
        "data": list(range(10)),
        "ratios": [1, 1, 1, 0],
        "num_partitions": None,
        "shuffle": False,
        "seed": 0,
        "drop_last": True,
        "even_divisible": True,
    },
    [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9], []],
]


TEST_CASE_14 = [
    {
        "data": list(range(52)),
        "ratios": [3, 1, 3, 8],
        "num_partitions": None,
        "shuffle": False,
        "seed": 0,
        "drop_last": True,
        "even_divisible": True,
    },
    # the largest rounding deficit is in the second partition, not the first,
    # so the leftover item goes there rather than to partition 0
    [list(range(10)), list(range(10, 14)), list(range(14, 24)), list(range(24, 52))],
]


class TestPartitionDataset(unittest.TestCase):
    @parameterized.expand(
        [
            TEST_CASE_1,
            TEST_CASE_2,
            TEST_CASE_3,
            TEST_CASE_4,
            TEST_CASE_5,
            TEST_CASE_6,
            TEST_CASE_7,
            TEST_CASE_8,
            TEST_CASE_9,
            TEST_CASE_10,
            TEST_CASE_11,
            TEST_CASE_12,
            TEST_CASE_13,
            TEST_CASE_14,
        ]
    )
    def test_value(self, input_param, result):
        self.assertListEqual(partition_dataset(**input_param), result)

    @parameterized.expand(TEST_CASES_ROUNDING)
    def test_ratios_rounding(self, input_param, result):
        """Items left over after rounding go to the partitions that were rounded down the most.

        Args:
            input_param: keyword arguments for ``partition_dataset``.
            result: the expected partitions.
        """
        self.assertListEqual(partition_dataset(**input_param), result)

    @parameterized.expand([(52, [3, 1, 3, 8], [10, 4, 10, 28]), (80, [9, 9, 3, 9, 2], [23, 23, 7, 22, 5])])
    def test_ratios_rounding_sizes(self, data_len, ratios, expected_sizes):
        """When rounding gives too few or too many items, adjust the partitions with the largest errors.

        Args:
            data_len: number of items to partition.
            ratios: the partition ratios.
            expected_sizes: partition sizes after correcting the rounded total.
        """
        partitions = partition_dataset(range(data_len), ratios=ratios)
        self.assertListEqual([len(part) for part in partitions], expected_sizes)

    @parameterized.expand(
        [(63, [0.8, 0.1, 0.1]), (303, [0.7, 0.15, 0.15]), (32, [1, 1, 1, 1, 1]), (80, [0, 9, 9, 3, 9, 2, 0])]
    )
    def test_ratios_assign_every_item(self, max_len, ratios):
        """Check coverage and rounding across dataset lengths, including splits with a zero ratio.

        Args:
            max_len: largest number of items to partition.
            ratios: the partition ratios.
        """
        total_ratio = sum(ratios)
        for data_len in range(max_len + 1):
            with self.subTest(data_len=data_len):
                data = list(range(data_len))
                partitions = partition_dataset(data, ratios=ratios, shuffle=True, seed=0)
                self.assertListEqual(sorted(i for part in partitions for i in part), data)
                sizes = [len(part) for part in partitions]
                exact = [r / total_ratio * data_len for r in ratios]
                for size, share in zip(sizes, exact, strict=True):
                    self.assertLess(abs(size - share), 1)
                rounded = [int(share + 0.5) for share in exact]
                if sum(rounded) == data_len:
                    self.assertListEqual(sizes, rounded)


if __name__ == "__main__":
    unittest.main()
