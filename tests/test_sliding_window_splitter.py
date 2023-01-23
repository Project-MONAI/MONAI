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
from torch.nn.functional import pad
from torch.testing import assert_close
from parameterized import parameterized

from monai.inferers import SlidingWindowSplitter
from monai.utils import PatchKeys, set_determinism


set_determinism(0)

TENSOR_4x4 = torch.randint(low=0, high=255, size=(2, 3, 4, 4), dtype=torch.float32)

TEST_CASE_0 = [
    TENSOR_4x4,
    {"patch_size": (2, 2), "batch_size": 1, "overlap": 0.0},
    [
        TENSOR_4x4[..., :2, :2],
        TENSOR_4x4[..., :2, 2:],
        TENSOR_4x4[..., 2:, :2],
        TENSOR_4x4[..., 2:, 2:],
    ],
]

TEST_CASE_1 = [
    TENSOR_4x4,
    {"patch_size": (3, 3), "batch_size": 1, "overlap": 0.0},
    [
        TENSOR_4x4[..., :3, :3],
        pad(TENSOR_4x4[..., :3, 3:], (0, 2)),
        pad(TENSOR_4x4[..., 3:, :3], (0, 0, 0, 2)),
        pad(TENSOR_4x4[..., 3:, 3:], (0, 2, 0, 2)),
    ],
]

TEST_CASE_2 = [
    TENSOR_4x4,
    {"patch_size": (2, 2), "batch_size": 1, "overlap": (0.5, 0.5)},
    [
        TENSOR_4x4[..., 0:2, 0:2],
        TENSOR_4x4[..., 0:2, 1:3],
        TENSOR_4x4[..., 0:2, 2:4],
        TENSOR_4x4[..., 1:3, 0:2],
        TENSOR_4x4[..., 1:3, 1:3],
        TENSOR_4x4[..., 1:3, 2:4],
        TENSOR_4x4[..., 2:4, 0:2],
        TENSOR_4x4[..., 2:4, 1:3],
        TENSOR_4x4[..., 2:4, 2:4],
    ],
]

TEST_CASE_3 = [
    TENSOR_4x4,
    {"patch_size": (3, 3), "batch_size": 1, "overlap": 2.0 / 3.0},
    [
        TENSOR_4x4[..., :3, :3],
        TENSOR_4x4[..., :3, 1:],
        TENSOR_4x4[..., 1:, :3],
        TENSOR_4x4[..., 1:, 1:],
    ],
]


class SlidingWindowSplitterTests(unittest.TestCase):
    @parameterized.expand(
        [
            TEST_CASE_0,
            TEST_CASE_1,
            TEST_CASE_2,
            TEST_CASE_3,
        ]
    )
    def test_split_patches(self, image, arguments, expected):
        patches = SlidingWindowSplitter(**arguments)(image)
        patches = list(patches)
        self.assertEqual(len(patches), len(expected))
        for p, e in zip(patches, expected):
            assert_close(p, e)


if __name__ == "__main__":
    unittest.main()
