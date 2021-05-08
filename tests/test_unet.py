# Copyright 2020 - 2021 MONAI Consortium
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

from monai.networks import eval_mode
from monai.networks.layers import Act, Norm
from monai.networks.nets import UNet
from tests.utils import test_script_save

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASE_0 = [  # single channel 2D, batch 16, no residual
    {
        "dimensions": 2,
        "in_channels": 1,
        "out_channels": 3,
        "channels": (16, 32, 64),
        "strides": (2, 2),
        "num_res_units": 0,
    },
    (16, 1, 32, 32),
    (16, 3, 32, 32),
]

TEST_CASE_1 = [  # single channel 2D, batch 16
    {
        "dimensions": 2,
        "in_channels": 1,
        "out_channels": 3,
        "channels": (16, 32, 64),
        "strides": (2, 2),
        "num_res_units": 1,
    },
    (16, 1, 32, 32),
    (16, 3, 32, 32),
]

TEST_CASE_2 = [  # single channel 3D, batch 16
    {
        "dimensions": 3,
        "in_channels": 1,
        "out_channels": 3,
        "channels": (16, 32, 64),
        "strides": (2, 2),
        "num_res_units": 1,
    },
    (16, 1, 32, 24, 48),
    (16, 3, 32, 24, 48),
]

TEST_CASE_3 = [  # 4-channel 3D, batch 16
    {
        "dimensions": 3,
        "in_channels": 4,
        "out_channels": 3,
        "channels": (16, 32, 64),
        "strides": (2, 2),
        "num_res_units": 1,
    },
    (16, 4, 32, 64, 48),
    (16, 3, 32, 64, 48),
]

TEST_CASE_4 = [  # 4-channel 3D, batch 16, batch normalization
    {
        "dimensions": 3,
        "in_channels": 4,
        "out_channels": 3,
        "channels": (16, 32, 64),
        "strides": (2, 2),
        "num_res_units": 1,
        "norm": Norm.BATCH,
    },
    (16, 4, 32, 64, 48),
    (16, 3, 32, 64, 48),
]

TEST_CASE_5 = [  # 4-channel 3D, batch 16, LeakyReLU activation
    {
        "dimensions": 3,
        "in_channels": 4,
        "out_channels": 3,
        "channels": (16, 32, 64),
        "strides": (2, 2),
        "num_res_units": 1,
        "act": (Act.LEAKYRELU, {"negative_slope": 0.2}),
    },
    (16, 4, 32, 64, 48),
    (16, 3, 32, 64, 48),
]

TEST_CASE_6 = [  # 4-channel 3D, batch 16, LeakyReLU activation explicit
    {
        "dimensions": 3,
        "in_channels": 4,
        "out_channels": 3,
        "channels": (16, 32, 64),
        "strides": (2, 2),
        "num_res_units": 1,
        "act": (torch.nn.LeakyReLU, {"negative_slope": 0.2}),
    },
    (16, 4, 32, 64, 48),
    (16, 3, 32, 64, 48),
]

CASES = [TEST_CASE_0, TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6]


class TestUNET(unittest.TestCase):
    @parameterized.expand(CASES)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = UNet(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)

    def test_script(self):
        net = UNet(
            dimensions=2,
            in_channels=1,
            out_channels=3,
            channels=(16, 32, 64),
            strides=(2, 2),
            num_res_units=0,
            norm=("batch", {"track_running_stats": False}),
        )
        test_data = torch.randn(16, 1, 32, 32)
        test_script_save(net, test_data)


if __name__ == "__main__":
    unittest.main()
