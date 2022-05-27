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

import unittest

import numpy as np
from parameterized import parameterized

from monai.transforms.spatial.array import RandGridPatch
from tests.utils import TEST_NDARRAYS, assert_allclose

A = np.arange(16).repeat(3).reshape(4, 4, 3).transpose(2, 0, 1)
A11 = A[:, :2, :2]
A12 = A[:, :2, 2:]
A21 = A[:, 2:, :2]
A22 = A[:, 2:, 2:]

TEST_CASE_0 = [{"patch_size": (2, 2), "min_start_pos": 0, "max_start_pos": 0}, A, [A11, A12, A21, A22]]
TEST_CASE_1 = [{"patch_size": (2, 2), "min_start_pos": 0, "num_patches": 3}, A, [A11, A12, A21]]
TEST_CASE_2 = [
    {"patch_size": (2, 2), "min_start_pos": 0, "max_start_pos": 0, "num_patches": 5},
    A,
    [A11, A12, A21, A22, np.zeros((3, 2, 2))],
]
TEST_CASE_3 = [{"patch_size": (2, 2), "min_start_pos": 0, "max_start_pos": 0}, A, [A11, A12, A21, A22]]
TEST_CASE_4 = [{"patch_size": (2, 2)}, A, [A11, A12, A21, A22]]
TEST_CASE_5 = [{"patch_size": (2, 2), "min_start_pos": 2, "max_start_pos": 2}, A, [A22]]
TEST_CASE_6 = [{"patch_size": (2, 2), "min_start_pos": (0, 2), "max_start_pos": (0, 2)}, A, [A12, A22]]
TEST_CASE_7 = [{"patch_size": (2, 2), "min_start_pos": 1, "max_start_pos": 2}, A, [A22]]
TEST_CASE_8 = [
    {"patch_size": (2, 2), "min_start_pos": 0, "max_start_pos": 1, "num_patches": 1, "sort_key": "max"},
    A,
    [A[:, 1:3, 1:3]],
]
TEST_CASE_9 = [
    {
        "patch_size": (3, 3),
        "min_start_pos": -3,
        "max_start_pos": -1,
        "sort_key": "min",
        "num_patches": 1,
        "pad_opts": {"constant_values": 255},
    },
    A,
    [np.pad(A[:, :2, 1:], ((0, 0), (1, 0), (0, 0)), mode="constant", constant_values=255)],
]

TEST_SINGLE = []
for p in TEST_NDARRAYS:
    TEST_SINGLE.append([p, *TEST_CASE_0])
    TEST_SINGLE.append([p, *TEST_CASE_1])
    TEST_SINGLE.append([p, *TEST_CASE_2])
    TEST_SINGLE.append([p, *TEST_CASE_3])
    TEST_SINGLE.append([p, *TEST_CASE_4])
    TEST_SINGLE.append([p, *TEST_CASE_5])
    TEST_SINGLE.append([p, *TEST_CASE_6])
    TEST_SINGLE.append([p, *TEST_CASE_7])
    TEST_SINGLE.append([p, *TEST_CASE_8])
    TEST_SINGLE.append([p, *TEST_CASE_9])


class TestSlidingPatch(unittest.TestCase):
    @parameterized.expand(TEST_SINGLE)
    def test_split_patch_single_call(self, in_type, input_parameters, image, expected):
        input_image = in_type(image)
        splitter = RandGridPatch(seed=1234, **input_parameters)
        output = list(splitter(input_image))
        self.assertEqual(len(output), len(expected))
        for output_patch, expected_patch in zip(output, expected):
            assert_allclose(output_patch[0], expected_patch, type_test=False)


if __name__ == "__main__":
    unittest.main()
