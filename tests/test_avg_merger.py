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

import torch
from parameterized import parameterized

from monai.inferers import AvgMerger
from tests.utils import assert_allclose

TENSOR_4x4 = torch.randint(low=0, high=255, size=(2, 3, 4, 4), dtype=torch.float32)
TENSOR_4x4_WITH_NAN = TENSOR_4x4.clone()
TENSOR_4x4_WITH_NAN[..., 2:, 2:] = torch.nan

# no-overlapping 2x2
TEST_CASE_SAME_SIZE_0 = [
    TENSOR_4x4,
    [
        (TENSOR_4x4[..., :2, :2], (0, 0)),
        (TENSOR_4x4[..., :2, 2:], (0, 2)),
        (TENSOR_4x4[..., 2:, :2], (2, 0)),
        (TENSOR_4x4[..., 2:, 2:], (2, 2)),
    ],
    TENSOR_4x4,
]

# overlapping 2x2
TEST_CASE_SAME_SIZE_1 = [
    TENSOR_4x4,
    [
        (TENSOR_4x4[..., 0:2, 0:2], (0, 0)),
        (TENSOR_4x4[..., 0:2, 1:3], (0, 1)),
        (TENSOR_4x4[..., 0:2, 2:4], (0, 2)),
        (TENSOR_4x4[..., 1:3, 0:2], (1, 0)),
        (TENSOR_4x4[..., 1:3, 1:3], (1, 1)),
        (TENSOR_4x4[..., 1:3, 2:4], (1, 2)),
        (TENSOR_4x4[..., 2:4, 0:2], (2, 0)),
        (TENSOR_4x4[..., 2:4, 1:3], (2, 1)),
        (TENSOR_4x4[..., 2:4, 2:4], (2, 2)),
    ],
    TENSOR_4x4,
]

# overlapping 3x3 (non-divisible)
TEST_CASE_SAME_SIZE_2 = [
    TENSOR_4x4,
    [
        (TENSOR_4x4[..., :3, :3], (0, 0)),
        (TENSOR_4x4[..., :3, 1:], (0, 1)),
        (TENSOR_4x4[..., 1:, :3], (1, 0)),
        (TENSOR_4x4[..., 1:, 1:], (1, 1)),
    ],
    TENSOR_4x4,
]

#  overlapping 2x2 with NaN values
TEST_CASE_SAME_SIZE_3 = [
    TENSOR_4x4_WITH_NAN,
    [
        (TENSOR_4x4_WITH_NAN[..., 0:2, 0:2], (0, 0)),
        (TENSOR_4x4_WITH_NAN[..., 0:2, 1:3], (0, 1)),
        (TENSOR_4x4_WITH_NAN[..., 0:2, 2:4], (0, 2)),
        (TENSOR_4x4_WITH_NAN[..., 1:3, 0:2], (1, 0)),
        (TENSOR_4x4_WITH_NAN[..., 1:3, 1:3], (1, 1)),
        (TENSOR_4x4_WITH_NAN[..., 1:3, 2:4], (1, 2)),
        (TENSOR_4x4_WITH_NAN[..., 2:4, 0:2], (2, 0)),
        (TENSOR_4x4_WITH_NAN[..., 2:4, 1:3], (2, 1)),
        (TENSOR_4x4_WITH_NAN[..., 2:4, 2:4], (2, 2)),
    ],
    TENSOR_4x4_WITH_NAN,
]

# non-overlapping 2x2 with missing patch
TEST_CASE_SAME_SIZE_4 = [
    TENSOR_4x4,
    [(TENSOR_4x4[..., :2, :2], (0, 0)), (TENSOR_4x4[..., :2, 2:], (0, 2)), (TENSOR_4x4[..., 2:, :2], (2, 0))],
    TENSOR_4x4_WITH_NAN,
]


class AvgMergerTests(unittest.TestCase):
    @parameterized.expand(
        [
            TEST_CASE_SAME_SIZE_0,
            TEST_CASE_SAME_SIZE_1,
            TEST_CASE_SAME_SIZE_2,
            TEST_CASE_SAME_SIZE_3,
            TEST_CASE_SAME_SIZE_4,
        ]
    )
    def test_avg_merger_patches_same_size(self, image, patch_locations, expected):
        merger = AvgMerger()
        merger.initialize(output_shape=image.shape)
        for pl in patch_locations:
            merger.aggregate(pl[0], pl[1])
        output = merger.finalize()
        assert_allclose(output, expected)

    def test_avg_merger_non_initialized_error(self):
        with self.assertRaises(ValueError):
            merger = AvgMerger()
            merger.aggregate(torch.zeros(1, 3, 2, 2), (3, 3))

    def test_avg_merge_no_output_shape_error(self):
        with self.assertRaises(ValueError):
            merger = AvgMerger()
            merger.initialize()


if __name__ == "__main__":
    unittest.main()
