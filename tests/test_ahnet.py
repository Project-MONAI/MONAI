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

from monai.networks.blocks import FCN, MCFCN
from monai.networks.nets import AHNet

TEST_CASE_FCN_1 = [{"nout": 3}, torch.randn(5, 3, 64, 64), (5, 3, 64, 64)]
TEST_CASE_FCN_2 = [{"nout": 2}, torch.randn(5, 3, 64, 64), (5, 2, 64, 64)]
TEST_CASE_FCN_3 = [{"nout": 1, "upsample_mode": "interpolate"}, torch.randn(5, 3, 64, 64), (5, 1, 64, 64)]

TEST_CASE_MCFCN_1 = [{"nout": 3, "nin": 8}, torch.randn(5, 8, 64, 64), (5, 3, 64, 64)]
TEST_CASE_MCFCN_2 = [{"nout": 2, "nin": 1}, torch.randn(5, 1, 64, 64), (5, 2, 64, 64)]
TEST_CASE_MCFCN_3 = [{"nout": 1, "nin": 2, "upsample_mode": "interpolate"}, torch.randn(5, 2, 64, 64), (5, 1, 64, 64)]

TEST_CASE_AHNET_1 = [{"upsample_mode": "interpolate"}, torch.randn(3, 1, 128, 128, 64), (3, 1, 128, 128, 64)]
TEST_CASE_AHNET_2 = [{"upsample_mode": "transpose"}, torch.randn(2, 1, 128, 128, 64), (2, 1, 128, 128, 64)]


class TestFCN(unittest.TestCase):
    @parameterized.expand([TEST_CASE_FCN_1, TEST_CASE_FCN_2, TEST_CASE_FCN_3])
    def test_fcn_shape(self, input_param, input_data, expected_shape):
        net = FCN(**input_param)
        net.eval()
        with torch.no_grad():
            result = net.forward(input_data)
            self.assertEqual(result.shape, expected_shape)


class TestMCFCN(unittest.TestCase):
    @parameterized.expand([TEST_CASE_MCFCN_1, TEST_CASE_MCFCN_2, TEST_CASE_MCFCN_3])
    def test_mcfcn_shape(self, input_param, input_data, expected_shape):
        net = MCFCN(**input_param)
        net.eval()
        with torch.no_grad():
            result = net.forward(input_data)
            self.assertEqual(result.shape, expected_shape)


class TestAHNET(unittest.TestCase):
    @parameterized.expand([TEST_CASE_AHNET_1, TEST_CASE_AHNET_2])
    def test_fcn_shape(self, input_param, input_data, expected_shape):
        net = AHNet(**input_param)
        net.eval()
        with torch.no_grad():
            result = net.forward(input_data)
            self.assertEqual(result.shape, expected_shape)
