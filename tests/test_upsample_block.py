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

import torch
from monai.networks.blocks import UpSample
from parameterized import parameterized

TEST_CASES = [
    [{"spatial_dims": 2, "in_channels": 4}, torch.randn(7, 4, 32, 48), (7, 4, 64, 96)],  # 4-channel 2D, batch 7
    [
        {"spatial_dims": 1, "in_channels": 4, "out_channels": 3},
        torch.randn(16, 4, 63),
        (16, 3, 126),
    ],  # 4-channel 1D, batch 16
    [
        {"spatial_dims": 1, "in_channels": 4, "out_channels": 8, "with_conv": True, "align_corners": False},
        torch.randn(16, 4, 20),
        (16, 8, 40),
    ],  # 4-channel 1D, batch 16
    [
        {"spatial_dims": 3, "in_channels": 4, "mode": "bilinear"},
        torch.randn(16, 4, 32, 24, 48),
        (16, 4, 64, 48, 96),
    ],  # 4-channel 3D, batch 16
    [
        {"spatial_dims": 3, "in_channels": 1, "with_conv": False, "scale_factor": 3, "align_corners": False},
        torch.randn(16, 1, 10, 15, 20),
        (16, 1, 30, 45, 60),
    ],  # 1-channel 3D, batch 16
]

TEST_CASES_EQ = []
for s in range(1, 5):
    expected_shape = (16, 5, 4 * s, 5 * s, 6 * s)
    for t in (False, True):
        test_case = [
            {
                "spatial_dims": 3,
                "in_channels": 3,
                "out_channels": 5,
                "with_conv": t,
                "scale_factor": s,
                "align_corners": True,
            },
            torch.randn(16, 3, 4, 5, 6),
        ]
        test_case.append(expected_shape)
        TEST_CASES_EQ.append(test_case)


class TestUpsample(unittest.TestCase):
    @parameterized.expand(TEST_CASES + TEST_CASES_EQ)
    def test_shape(self, input_param, input_data, expected_shape):
        net = UpSample(**input_param)
        net.eval()
        with torch.no_grad():
            result = net(input_data)
            self.assertEqual(result.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
