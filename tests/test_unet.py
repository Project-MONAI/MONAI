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

from monai.networks.nets.unet import UNet

TEST_CASE_1 = [  # single channel 2D, batch 16
    {
        'dimensions': 2,
        'in_channels': 1,
        'num_classes': 3,
        'channels': (16, 32, 64),
        'strides': (2, 2),
        'num_res_units': 1,
    },
    torch.randn(16, 1, 32, 32),
    (16, 32, 32),
]

TEST_CASE_2 = [  # single channel 3D, batch 16
    {
        'dimensions': 3,
        'in_channels': 1,
        'num_classes': 3,
        'channels': (16, 32, 64),
        'strides': (2, 2),
        'num_res_units': 1,
    },
    torch.randn(16, 1, 32, 24, 48),
    (16, 32, 24, 48),
]

TEST_CASE_3 = [  # 4-channel 3D, batch 16
    {
        'dimensions': 3,
        'in_channels': 4,
        'num_classes': 3,
        'channels': (16, 32, 64),
        'strides': (2, 2),
        'num_res_units': 1,
    },
    torch.randn(16, 4, 32, 64, 48),
    (16, 32, 64, 48),
]


class TestUNET(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_shape(self, input_param, input_data, expected_shape):
        result = UNet(**input_param).forward(input_data)[1]
        self.assertEqual(result.shape, expected_shape)


if __name__ == '__main__':
    unittest.main()
