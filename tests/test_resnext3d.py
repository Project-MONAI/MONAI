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

from monai.networks.nets.resnext3d import resnext50, resnext101, resnext152


TEST_CASE_1 = [  # 4-channel 3D, batch 16
    {
        'num_input_channel': 2,
        'num_classes': 3
    },
    torch.randn(16, 2, 32, 64, 48),
    (16, 3)
]


class TestRESNEXT3D(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1])
    def test_50_shape(self, input_param, input_data, expected_shape):
        net = resnext50(**input_param)
        net.eval()
        with torch.no_grad():
            result = net.forward(input_data)
            self.assertEqual(result.shape, expected_shape)

    @parameterized.expand([TEST_CASE_1])
    def test_101_shape(self, input_param, input_data, expected_shape):
        net = resnext101(**input_param)
        net.eval()
        with torch.no_grad():
            result = net.forward(input_data)
            self.assertEqual(result.shape, expected_shape)

    @parameterized.expand([TEST_CASE_1])
    def test_152_shape(self, input_param, input_data, expected_shape):
        net = resnext152(**input_param)
        net.eval()
        with torch.no_grad():
            result = net.forward(input_data)
            self.assertEqual(result.shape, expected_shape)
