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

from monai.transforms.spatial.array import GridPatch
from tests.utils import TEST_NDARRAYS, assert_allclose

A = np.arange(16).repeat(3).reshape(4, 4, 3).transpose(2, 0, 1)
A11 = A[:, :2, :2]
A12 = A[:, :2, 2:]
A21 = A[:, 2:, :2]
A22 = A[:, 2:, 2:]

TEST_CASE_0 = [{"patch_size": (2, 2)}, A, [A11, A12, A21, A22]]
TEST_CASE_1 = [{"patch_size": (2, 2), "fix_num_patches": 3}, A, [A11, A12, A21]]
TEST_CASE_2 = [{"patch_size": (2, 2), "fix_num_patches": 5}, A, [A11, A12, A21, A22, np.zeros((3, 2, 2))]]
TEST_CASE_3 = [{"patch_size": (2, 2), "start_pos": (0, 0)}, A, [A11, A12, A21, A22]]
TEST_CASE_4 = [{"patch_size": (2, 2), "start_pos": (0, 0)}, A, [A11, A12, A21, A22]]
TEST_CASE_5 = [{"patch_size": (2, 2), "start_pos": (2, 2)}, A, [A22]]
TEST_CASE_6 = [{"patch_size": (2, 2), "start_pos": (0, 2)}, A, [A12, A22]]
TEST_CASE_7 = [{"patch_size": (2, 2), "start_pos": (2, 0)}, A, [A21, A22]]
TEST_CASE_8 = [{"patch_size": (2, 2), "fix_num_patches": 3, "sort_key": "max"}, A, [A22, A21, A12]]
TEST_CASE_9 = [{"patch_size": (2, 2), "fix_num_patches": 4, "sort_key": "min"}, A, [A11, A12, A21, A22]]
TEST_CASE_10 = [{"patch_size": (2, 2), "overlap": 0.5, "fix_num_patches": 3}, A, [A11, A[:, :2, 1:3], A12]]
TEST_CASE_11 = [
    {"patch_size": (3, 3), "fix_num_patches": 2, "pad_opts": {"constant_values": 255}},
    A,
    [A[:, :3, :3], np.pad(A[:, :3, 3:], ((0, 0), (0, 0), (0, 2)), mode="constant", constant_values=255)],
]
TEST_CASE_12 = [
    {"patch_size": (3, 3), "start_pos": (-2, -2), "fix_num_patches": 2},
    A,
    [np.zeros((3, 3, 3)), np.pad(A[:, :1, 1:4], ((0, 0), (2, 0), (0, 0)), mode="constant")],
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
    TEST_SINGLE.append([p, *TEST_CASE_10])
    TEST_SINGLE.append([p, *TEST_CASE_11])
    TEST_SINGLE.append([p, *TEST_CASE_12])


class TestSlidingPatch(unittest.TestCase):
    @parameterized.expand(TEST_SINGLE)
    def test_split_patch_single_call(self, in_type, input_parameters, image, expected):
        input_image = in_type(image)
        splitter = GridPatch(**input_parameters)
        output = list(splitter(input_image))
        self.assertEqual(len(output), len(expected))
        for output_patch, expected_patch in zip(output, expected):
            assert_allclose(output_patch[0], expected_patch, type_test=False)


if __name__ == "__main__":
    unittest.main()
