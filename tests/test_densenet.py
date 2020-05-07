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

from monai.networks.nets import densenet121, densenet169, densenet201, densenet264


TEST_CASE_1 = [  # 4-channel 3D, batch 16
    {"spatial_dims": 3, "in_channels": 2, "out_channels": 3},
    torch.randn(16, 2, 32, 64, 48),
    (16, 3),
]


class TestDENSENET(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1])
    def test_121_shape(self, input_param, input_data, expected_shape):
        net = densenet121(**input_param)
        net.eval()
        with torch.no_grad():
            result = net.forward(input_data)
            self.assertEqual(result.shape, expected_shape)

    @parameterized.expand([TEST_CASE_1])
    def test_169_shape(self, input_param, input_data, expected_shape):
        net = densenet169(**input_param)
        net.eval()
        with torch.no_grad():
            result = net.forward(input_data)
            self.assertEqual(result.shape, expected_shape)

    @parameterized.expand([TEST_CASE_1])
    def test_201_shape(self, input_param, input_data, expected_shape):
        net = densenet201(**input_param)
        net.eval()
        with torch.no_grad():
            result = net.forward(input_data)
            self.assertEqual(result.shape, expected_shape)

    @parameterized.expand([TEST_CASE_1])
    def test_264_shape(self, input_param, input_data, expected_shape):
        net = densenet264(**input_param)
        net.eval()
        with torch.no_grad():
            result = net.forward(input_data)
            self.assertEqual(result.shape, expected_shape)
