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
from parameterized import parameterized

from monai.networks.nets import BasicUNet

CASES_2D = []
for mode in ["pixelshuffle", "nontrainable", "deconv"]:
    for d1 in range(17, 64, 14):
        for d2 in range(63, 18, -21):
            in_channels, out_channels = 2, 3
            CASES_2D.append(
                [
                    {
                        "dimensions": 2,
                        "in_channels": in_channels,
                        "out_channels": out_channels,
                        "upsample": mode,
                    },
                    torch.randn(2, in_channels, d1, d2),
                    (2, out_channels, d1, d2),
                ]
            )
CASES_3D = [
    [  # single channel 3D, batch 2
        {
            "dimensions": 3,
            "in_channels": 1,
            "out_channels": 2,
            "features": (16, 20, 21, 22, 23, 11),
            "upsample": "pixelshuffle",
        },
        torch.randn(2, 1, 16, 17, 18),
        (2, 2, 16, 17, 18),
    ],
    [  # 2-channel 3D, batch 3
        {
            "dimensions": 3,
            "in_channels": 2,
            "out_channels": 7,
            "features": (14, 15, 16, 17, 18, 11),
            "upsample": "deconv",
        },
        torch.randn(3, 2, 16, 17, 18),
        (3, 7, 16, 17, 18),
    ],
    [  # 4-channel 3D, batch 5
        {
            "dimensions": 3,
            "in_channels": 4,
            "out_channels": 2,
            "features": (14, 15, 16, 17, 18, 10),
            "upsample": "nontrainable",
        },
        torch.randn(5, 4, 128, 128, 16),
        (5, 2, 128, 128, 16),
    ],
]


class TestBaseUNET(unittest.TestCase):
    @parameterized.expand(CASES_2D + CASES_3D)
    def test_shape(self, input_param, input_data, expected_shape):
        net = BasicUNet(**input_param)
        net.eval()
        with torch.no_grad():
            result = net.forward(input_data)
            self.assertEqual(result.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
