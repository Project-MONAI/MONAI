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
from torch.nn.functional import avg_pool2d

from monai.inferers import AvgMerger, PatchInferer, SlidingWindowSplitter
from tests.utils import assert_allclose

TENSOR_4x4 = torch.randint(low=0, high=255, size=(2, 3, 4, 4), dtype=torch.float32)
TENSOR_2x2 = avg_pool2d(TENSOR_4x4, 2, 2)

# no-overlapping 2x2 patches
TEST_CASE_0_TENSOR = [
    TENSOR_4x4,
    dict(splitter=SlidingWindowSplitter(patch_size=(2, 2)), merger=AvgMerger()),
    lambda x: x,
    TENSOR_4x4,
]

# no-overlapping 2x2 patches using all default parameters (except for splitter)
TEST_CASE_1_TENSOR = [TENSOR_4x4, dict(splitter=SlidingWindowSplitter(patch_size=(2, 2))), lambda x: x, TENSOR_4x4]

# divisible batch_size
TEST_CASE_2_TENSOR = [
    TENSOR_4x4,
    dict(splitter=SlidingWindowSplitter(patch_size=(2, 2)), merger=AvgMerger(), batch_size=2),
    lambda x: x,
    TENSOR_4x4,
]

# non-divisible batch_size
TEST_CASE_3_TENSOR = [
    TENSOR_4x4,
    dict(splitter=SlidingWindowSplitter(patch_size=(2, 2)), merger=AvgMerger(), batch_size=3),
    lambda x: x,
    TENSOR_4x4,
]

# patches that are already split (Splitter should be None)
TEST_CASE_4_SPLIT_LIST = [
    [
        (TENSOR_4x4[..., :2, :2], (0, 0)),
        (TENSOR_4x4[..., :2, 2:], (0, 2)),
        (TENSOR_4x4[..., 2:, :2], (2, 0)),
        (TENSOR_4x4[..., 2:, 2:], (2, 2)),
    ],
    dict(splitter=None, merger=AvgMerger(output_shape=(2, 3, 4, 4))),
    lambda x: x,
    TENSOR_4x4,
]


# using all default parameters (patches are already split)
TEST_CASE_5_SPLIT_LIST = [
    [
        (TENSOR_4x4[..., :2, :2], (0, 0)),
        (TENSOR_4x4[..., :2, 2:], (0, 2)),
        (TENSOR_4x4[..., 2:, :2], (2, 0)),
        (TENSOR_4x4[..., 2:, 2:], (2, 2)),
    ],
    dict(merger=AvgMerger(output_shape=(2, 3, 4, 4))),
    lambda x: x,
    TENSOR_4x4,
]

# output smaller than input patches
TEST_CASE_6_SMALLER = [
    TENSOR_4x4,
    dict(splitter=SlidingWindowSplitter(patch_size=(2, 2)), merger=AvgMerger()),
    lambda x: torch.mean(x, dim=(-1, -2), keepdim=True),
    TENSOR_2x2,
]


# list of tensor output
TEST_CASE_0_LIST_TENSOR = [
    TENSOR_4x4,
    dict(splitter=SlidingWindowSplitter(patch_size=(2, 2)), merger=(AvgMerger(), AvgMerger())),
    lambda x: (x, x),
    (TENSOR_4x4, TENSOR_4x4),
]

# list of tensor output
TEST_CASE_0_DICT = [
    TENSOR_4x4,
    dict(splitter=SlidingWindowSplitter(patch_size=(2, 2)), merger=AvgMerger()),
    lambda x: {"model_output": x},
    {"model_output": TENSOR_4x4},
]

# ----------------------------------------------------------------------------
# Error test cases
# ----------------------------------------------------------------------------
# invalid splitter: not callable
TEST_CASE_ERROR_0 = [None, dict(splitter=1), TypeError]
# invalid splitter: not callable
TEST_CASE_ERROR_1 = [None, dict(splitter=lambda x: x, merger=lambda x: x), TypeError]
TEST_CASE_ERROR_2 = [None, dict(splitter=lambda x: x, merger=[AvgMerger(), None]), TypeError]
# invalid preprocessing
TEST_CASE_ERROR_3 = [None, dict(splitter=lambda x: x, pre_processing=1), TypeError]
# invalid postprocessing
TEST_CASE_ERROR_4 = [None, dict(splitter=lambda x: x, post_processing=1), TypeError]
# provide splitter when data is already split (splitter is not None)
TEST_CASE_ERROR_5 = [
    [
        (TENSOR_4x4[..., :2, :2], (0, 0)),
        (TENSOR_4x4[..., :2, 2:], (0, 2)),
        (TENSOR_4x4[..., 2:, :2], (2, 0)),
        (TENSOR_4x4[..., 2:, 2:], (2, 2)),
    ],
    dict(splitter=lambda x: x),
    ValueError,
]


class PatchInfererTests(unittest.TestCase):
    @parameterized.expand(
        [
            TEST_CASE_0_TENSOR,
            TEST_CASE_1_TENSOR,
            TEST_CASE_2_TENSOR,
            TEST_CASE_3_TENSOR,
            TEST_CASE_4_SPLIT_LIST,
            TEST_CASE_5_SPLIT_LIST,
            TEST_CASE_6_SMALLER,
        ]
    )
    def test_patch_inferer_tensor(self, inputs, arguments, network, expected):
        inferer = PatchInferer(**arguments)
        output = inferer(inputs=inputs, network=network)
        assert_allclose(output, expected)

    @parameterized.expand([TEST_CASE_0_LIST_TENSOR])
    def test_patch_inferer_list_tensor(self, inputs, arguments, network, expected):
        inferer = PatchInferer(**arguments)
        output = inferer(inputs=inputs, network=network)
        for out, exp in zip(output, expected):
            assert_allclose(out, exp)

    @parameterized.expand([TEST_CASE_0_DICT])
    def test_patch_inferer_dict(self, inputs, arguments, network, expected):
        inferer = PatchInferer(**arguments)
        output = inferer(inputs=inputs, network=network)
        for k in expected:
            assert_allclose(output[k], expected[k])

    @parameterized.expand(
        [TEST_CASE_ERROR_0, TEST_CASE_ERROR_1, TEST_CASE_ERROR_2, TEST_CASE_ERROR_3, TEST_CASE_ERROR_4]
    )
    def test_patch_inferer_errors(self, inputs, arguments, expected_error):
        with self.assertRaises(expected_error):
            PatchInferer(**arguments)
            inferer = PatchInferer(**arguments)
            inferer(inputs=inputs, network=lambda x: x)


if __name__ == "__main__":
    unittest.main()
