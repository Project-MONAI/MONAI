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
from torch.nn.functional import pad

from monai.inferers import AvgMerger, StreamingSlabAvgMerger
from tests.test_utils import assert_allclose

TENSOR_4x4 = torch.randint(low=0, high=255, size=(2, 3, 4, 4), dtype=torch.float32)
TENSOR_4x4_WITH_NAN = TENSOR_4x4.clone()
TENSOR_4x4_WITH_NAN[..., 2:, 2:] = float("nan")

# no-overlapping 2x2
TEST_CASE_0_DEFAULT_DTYPE = [
    dict(merged_shape=TENSOR_4x4.shape),
    [
        (TENSOR_4x4[..., :2, :2], (0, 0)),
        (TENSOR_4x4[..., :2, 2:], (0, 2)),
        (TENSOR_4x4[..., 2:, :2], (2, 0)),
        (TENSOR_4x4[..., 2:, 2:], (2, 2)),
    ],
    TENSOR_4x4,
]

# overlapping 2x2
TEST_CASE_1_DEFAULT_DTYPE = [
    dict(merged_shape=TENSOR_4x4.shape),
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
TEST_CASE_2_DEFAULT_DTYPE = [
    dict(merged_shape=TENSOR_4x4.shape),
    [
        (TENSOR_4x4[..., :3, :3], (0, 0)),
        (TENSOR_4x4[..., :3, 1:], (0, 1)),
        (TENSOR_4x4[..., 1:, :3], (1, 0)),
        (TENSOR_4x4[..., 1:, 1:], (1, 1)),
    ],
    TENSOR_4x4,
]

#  overlapping 2x2 with NaN values
TEST_CASE_3_DEFAULT_DTYPE = [
    dict(merged_shape=TENSOR_4x4_WITH_NAN.shape),
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
TEST_CASE_4_DEFAULT_DTYPE = [
    dict(merged_shape=TENSOR_4x4.shape),
    [(TENSOR_4x4[..., :2, :2], (0, 0)), (TENSOR_4x4[..., :2, 2:], (0, 2)), (TENSOR_4x4[..., 2:, :2], (2, 0))],
    TENSOR_4x4_WITH_NAN,
]

# with value_dtype set to half precision
TEST_CASE_5_VALUE_DTYPE = [
    dict(merged_shape=TENSOR_4x4.shape, value_dtype=torch.float16),
    [
        (TENSOR_4x4[..., :2, :2], (0, 0)),
        (TENSOR_4x4[..., :2, 2:], (0, 2)),
        (TENSOR_4x4[..., 2:, :2], (2, 0)),
        (TENSOR_4x4[..., 2:, 2:], (2, 2)),
    ],
    TENSOR_4x4,
]
# with count_dtype set to int32
TEST_CASE_6_COUNT_DTYPE = [
    dict(merged_shape=TENSOR_4x4.shape, count_dtype=torch.int32),
    [
        (TENSOR_4x4[..., :2, :2], (0, 0)),
        (TENSOR_4x4[..., :2, 2:], (0, 2)),
        (TENSOR_4x4[..., 2:, :2], (2, 0)),
        (TENSOR_4x4[..., 2:, 2:], (2, 2)),
    ],
    TENSOR_4x4,
]
# with both value_dtype, count_dtype set to double precision
TEST_CASE_7_COUNT_VALUE_DTYPE = [
    dict(merged_shape=TENSOR_4x4.shape, value_dtype=torch.float64, count_dtype=torch.float64),
    [
        (TENSOR_4x4[..., :2, :2], (0, 0)),
        (TENSOR_4x4[..., :2, 2:], (0, 2)),
        (TENSOR_4x4[..., 2:, :2], (2, 0)),
        (TENSOR_4x4[..., 2:, 2:], (2, 2)),
    ],
    TENSOR_4x4,
]

# shape larger than what is covered by patches
TEST_CASE_8_LARGER_SHAPE = [
    dict(merged_shape=(2, 3, 4, 6)),
    [
        (TENSOR_4x4[..., :2, :2], (0, 0)),
        (TENSOR_4x4[..., :2, 2:], (0, 2)),
        (TENSOR_4x4[..., 2:, :2], (2, 0)),
        (TENSOR_4x4[..., 2:, 2:], (2, 2)),
    ],
    pad(TENSOR_4x4, (0, 2), value=float("nan")),
]


class AvgMergerTests(unittest.TestCase):
    @parameterized.expand(
        [
            TEST_CASE_0_DEFAULT_DTYPE,
            TEST_CASE_1_DEFAULT_DTYPE,
            TEST_CASE_2_DEFAULT_DTYPE,
            TEST_CASE_3_DEFAULT_DTYPE,
            TEST_CASE_4_DEFAULT_DTYPE,
            TEST_CASE_5_VALUE_DTYPE,
            TEST_CASE_6_COUNT_DTYPE,
            TEST_CASE_7_COUNT_VALUE_DTYPE,
            TEST_CASE_8_LARGER_SHAPE,
        ]
    )
    def test_avg_merger_patches(self, arguments, patch_locations, expected):
        merger = AvgMerger(**arguments)
        for pl in patch_locations:
            merger.aggregate(pl[0], pl[1])
        output = merger.finalize()
        if "value_dtype" in arguments:
            self.assertTrue(merger.get_values().dtype, arguments["value_dtype"])
        if "count_dtype" in arguments:
            self.assertTrue(merger.get_counts().dtype, arguments["count_dtype"])
        # check for multiple call of finalize
        self.assertIs(output, merger.finalize())
        # check if the result is matching the expectation
        assert_allclose(output, expected)

    def test_avg_merger_finalized_error(self):
        with self.assertRaises(ValueError):
            merger = AvgMerger(merged_shape=(1, 3, 2, 3))
            merger.finalize()
            merger.aggregate(torch.zeros(1, 3, 2, 2), (3, 3))

    def test_avg_merge_none_merged_shape_error(self):
        with self.assertRaises(ValueError):
            AvgMerger(merged_shape=None)


class TestStreamingSlabAvgMerger(unittest.TestCase):
    def test_streaming_slab_merger_basic(self):
        tensor = torch.randn(2, 3, 16, 16, 16)
        roi = (8, 8, 8)
        merger_avg = AvgMerger(tensor.shape)
        merger_slab = StreamingSlabAvgMerger(tensor.shape, roi, slab_dim=0)

        for z in (0, 4, 8):
            for y in (0, 4, 8):
                for x in (0, 4, 8):
                    loc = (z, y, x)
                    patch = tensor[:, :, z : z + 8, y : y + 8, x : x + 8]
                    merger_avg.aggregate(patch, loc)
                    merger_slab.aggregate(patch, loc)

        res_avg = merger_avg.finalize()
        res_slab = merger_slab.finalize()
        diff = (res_avg - res_slab).abs().max().item()
        self.assertLess(diff, 1e-5)

    def test_streaming_slab_merger_spatial_dims(self):
        """Verify streaming slab merger operates correctly across all spatial dimensions."""
        tensor = torch.randn(1, 2, 16, 16, 16)
        roi = (8, 8, 8)
        merger_avg = AvgMerger(tensor.shape)
        for z in (0, 8):
            for y in (0, 8):
                for x in (0, 8):
                    loc = (z, y, x)
                    patch = tensor[:, :, z : z + 8, y : y + 8, x : x + 8]
                    merger_avg.aggregate(patch, loc)
        expected = merger_avg.finalize()

        for sdim in (0, 1, 2):
            merger_slab = StreamingSlabAvgMerger(tensor.shape, roi, slab_dim=sdim)
            # Sort locations by coordinate along sdim
            locs = [(z, y, x) for z in (0, 8) for y in (0, 8) for x in (0, 8)]
            locs.sort(key=lambda l: l[sdim])
            for loc in locs:
                z, y, x = loc
                patch = tensor[:, :, z : z + 8, y : y + 8, x : x + 8]
                merger_slab.aggregate(patch, loc)
            out = merger_slab.finalize()
            diff = (expected - out).abs().max().item()
            self.assertLess(diff, 1e-5)

    def test_streaming_slab_merger_finalized_error(self):
        """Verify aggregate raises ValueError when invoked on an already finalized merger."""
        merger = StreamingSlabAvgMerger(merged_shape=(1, 3, 4, 4), roi_size=(2, 2))
        merger.finalize()
        with self.assertRaises(ValueError):
            merger.aggregate(torch.zeros(1, 3, 2, 2), (0, 0))

    def test_streaming_slab_merger_uncovered_nan(self):
        """Uncovered locations should preserve NaN, matching AvgMerger semantics."""
        shape = (1, 1, 4, 6)
        roi = (2, 2)
        merger_avg = AvgMerger(shape)
        merger_slab = StreamingSlabAvgMerger(shape, roi, slab_dim=0)

        # Only aggregate first 4 columns, leaving columns 4..6 uncovered
        for z in (0, 2):
            for y in (0, 2):
                p = torch.ones(1, 1, 2, 2)
                merger_avg.aggregate(p, (z, y))
                merger_slab.aggregate(p, (z, y))

        res_avg = merger_avg.finalize()
        res_slab = merger_slab.finalize()

        # Both should have NaN in the uncovered regions
        self.assertTrue(torch.all(torch.isnan(res_avg[:, :, :, 4:])))
        self.assertTrue(torch.all(torch.isnan(res_slab[:, :, :, 4:])))
        # Covered regions should match
        self.assertTrue(torch.allclose(res_avg[:, :, :, :4], res_slab[:, :, :, :4]))

    def test_streaming_slab_merger_out_of_order_error(self):
        """Patches arriving with decreasing slab_dim coordinates must raise ValueError."""
        merger = StreamingSlabAvgMerger(merged_shape=(1, 1, 8, 8), roi_size=(4, 4), slab_dim=0)
        merger.aggregate(torch.ones(1, 1, 4, 4), (4, 0))
        with self.assertRaises(ValueError):
            merger.aggregate(torch.ones(1, 1, 4, 4), (0, 0))


if __name__ == "__main__":
    unittest.main()
