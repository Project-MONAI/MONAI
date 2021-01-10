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

import numpy as np
import torch
from parameterized import parameterized

from monai.losses.image_dissimilarity import LocalNormalizedCrossCorrelationLoss

TEST_CASES = [
    [
        {"in_channels": 3, "ndim": 3, "kernel_size": 3, "kernel_type": "rectangular"},
        {
            "input": torch.arange(0, 3, dtype=torch.float)[None, :, None, None, None].expand(1, 3, 3, 3, 3),
            "target": torch.arange(0, 3, dtype=torch.float)[None, :, None, None, None].expand(1, 3, 3, 3, 3),
        },
        -1.0,
    ],
    [
        {"in_channels": 3, "ndim": 2, "kernel_size": 3, "kernel_type": "rectangular"},
        {
            "input": torch.arange(0, 3, dtype=torch.float)[None, :, None, None].expand(1, 3, 3, 3),
            "target": torch.arange(0, 3, dtype=torch.float)[None, :, None, None].expand(1, 3, 3, 3),
        },
        -1.0,
    ],
    [
        {"in_channels": 3, "ndim": 2, "kernel_size": 3, "kernel_type": "triangular"},
        {
            "input": torch.arange(0, 3, dtype=torch.float)[None, :, None, None].expand(1, 3, 3, 3),
            "target": torch.arange(0, 3, dtype=torch.float)[None, :, None, None].expand(1, 3, 3, 3),
        },
        -1.0,
    ],
    [
        {"in_channels": 3, "ndim": 2, "kernel_size": 3, "kernel_type": "gaussian"},
        {
            "input": torch.arange(0, 3, dtype=torch.float)[None, :, None, None].expand(1, 3, 3, 3),
            "target": torch.arange(0, 3, dtype=torch.float)[None, :, None, None].expand(1, 3, 3, 3),
        },
        -1.0,
    ],
    [
        {"in_channels": 3, "ndim": 1, "kernel_size": 3, "kernel_type": "rectangular"},
        {
            "input": torch.arange(0, 3, dtype=torch.float)[None, :, None].expand(1, 3, 3),
            "target": torch.arange(0, 3, dtype=torch.float)[None, :, None].expand(1, 3, 3),
        },
        -1.0,
    ],
    [
        {"in_channels": 3, "ndim": 1, "kernel_size": 3, "kernel_type": "triangular"},
        {
            "input": torch.arange(0, 3, dtype=torch.float)[None, :, None].expand(1, 3, 3),
            "target": torch.arange(0, 3, dtype=torch.float)[None, :, None].expand(1, 3, 3),
        },
        -1.0,
    ],
    [
        {"in_channels": 3, "ndim": 1, "kernel_size": 3, "kernel_type": "gaussian"},
        {
            "input": torch.arange(0, 3, dtype=torch.float)[None, :, None].expand(1, 3, 3),
            "target": torch.arange(0, 3, dtype=torch.float)[None, :, None].expand(1, 3, 3),
        },
        -1.0,
    ],
    [
        {"in_channels": 3, "ndim": 1, "kernel_size": 3, "kernel_type": "gaussian", "reduction": "sum"},
        {
            "input": torch.arange(0, 3, dtype=torch.float)[None, :, None].expand(2, 3, 3),
            "target": torch.arange(0, 3, dtype=torch.float)[None, :, None].expand(2, 3, 3),
        },
        -6.0,
    ],
    [
        {"in_channels": 3, "ndim": 3, "kernel_size": 3, "kernel_type": "rectangular"},
        {
            "input": torch.arange(0, 3, dtype=torch.float)[None, :, None, None, None].expand(1, 3, 3, 3, 3),
            "target": torch.arange(0, 3, dtype=torch.float)[None, :, None, None, None].expand(1, 3, 3, 3, 3) ** 2,
        },
        -0.06062524,
    ],
    [
        {"in_channels": 3, "ndim": 3, "kernel_size": 5, "kernel_type": "triangular"},
        {
            "input": torch.arange(0, 3, dtype=torch.float)[None, :, None, None, None].expand(1, 3, 3, 3, 3),
            "target": torch.arange(0, 3, dtype=torch.float)[None, :, None, None, None].expand(1, 3, 3, 3, 3) ** 2,
        },
        -0.923356,
    ],
    [
        {"in_channels": 3, "ndim": 3, "kernel_size": 3, "kernel_type": "gaussian"},
        {
            "input": torch.arange(0, 3, dtype=torch.float)[None, :, None, None, None].expand(1, 3, 3, 3, 3),
            "target": torch.arange(0, 3, dtype=torch.float)[None, :, None, None, None].expand(1, 3, 3, 3, 3) ** 2,
        },
        -1.306177,
    ],
]


class TestLocalNormalizedCrossCorrelationLoss(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_shape(self, input_param, input_data, expected_val):
        result = LocalNormalizedCrossCorrelationLoss(**input_param).forward(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, rtol=1e-4)

    def test_ill_shape(self):
        loss = LocalNormalizedCrossCorrelationLoss(in_channels=3, ndim=3)
        # in_channel unmatch
        with self.assertRaisesRegex(AssertionError, ""):
            loss.forward(torch.ones((1, 2, 3, 3, 3), dtype=torch.float), torch.ones((1, 2, 3, 3, 3), dtype=torch.float))
        # ndim unmatch
        with self.assertRaisesRegex(AssertionError, ""):
            loss.forward(torch.ones((1, 3, 3, 3), dtype=torch.float), torch.ones((1, 3, 3, 3), dtype=torch.float))
        # input, target shape unmatch
        with self.assertRaisesRegex(AssertionError, ""):
            loss.forward(torch.ones((1, 3, 3, 3, 3), dtype=torch.float), torch.ones((1, 3, 4, 4, 4), dtype=torch.float))

    def test_ill_opts(self):
        input = torch.ones((1, 3, 3, 3, 3), dtype=torch.float)
        target = torch.ones((1, 3, 3, 3, 3), dtype=torch.float)
        with self.assertRaisesRegex(ValueError, ""):
            LocalNormalizedCrossCorrelationLoss(in_channels=3, kernel_type="unknown")(input, target)
        with self.assertRaisesRegex(ValueError, ""):
            LocalNormalizedCrossCorrelationLoss(in_channels=3, kernel_type=None)(input, target)
        with self.assertRaisesRegex(ValueError, ""):
            LocalNormalizedCrossCorrelationLoss(in_channels=3, kernel_size=4)(input, target)
        with self.assertRaisesRegex(ValueError, ""):
            LocalNormalizedCrossCorrelationLoss(in_channels=3, reduction="unknown")(input, target)
        with self.assertRaisesRegex(ValueError, ""):
            LocalNormalizedCrossCorrelationLoss(in_channels=3, reduction=None)(input, target)


if __name__ == "__main__":
    unittest.main()
