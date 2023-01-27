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
from torch.testing import assert_close
from parameterized import parameterized

from monai.inferers import AvgMerger, PatchInferer, SlidingWindowSplitter


TENSOR_4x4 = torch.randint(low=0, high=255, size=(2, 3, 4, 4), dtype=torch.float32)

# no-overlapping 2x2
TEST_CASE_1 = [
    TENSOR_4x4,
    dict(
        splitter=SlidingWindowSplitter(patch_size=(2, 2)),
        merger=AvgMerger(),
    ),
    lambda x: x,
    TENSOR_4x4,
]


class PatchInfererTests(unittest.TestCase):
    @parameterized.expand(
        [
            TEST_CASE_1,
        ]
    )
    def test_inference(self, inputs, arguments, network, expected):
        inferer = PatchInferer(**arguments)
        output = inferer(inputs=inputs, network=network)
        assert_close(output, expected, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
