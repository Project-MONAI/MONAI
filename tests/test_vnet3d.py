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

from monai.networks.nets.vnet3d import VNet3D


TEST_CASE_1 = [  # 4-channel 3D, batch 16
    {
        'num_input_channel': 1,
        'num_output_channel': 3
    },
    torch.randn(16, 1, 32, 64, 48),
    (16, 3, 32, 64, 48)
]


class TestVNET(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1])
    def test_3d_shape(self, input_param, input_data, expected_shape):
        net = VNet3D(**input_param)
        net.eval()
        with torch.no_grad():
            result = net.forward(input_data)
            self.assertEqual(result.shape, expected_shape)
