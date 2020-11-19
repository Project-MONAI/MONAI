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

from monai.networks.nets import (
    se_resnet50,
    se_resnet101,
    se_resnet152,
    se_resnext50_32x4d,
    se_resnext101_32x4d,
    senet154,
)
from tests.utils import test_pretrained_networks, test_script_save

device = "cuda" if torch.cuda.is_available() else "cpu"

net_args = {"spatial_dims": 3, "in_channels": 2, "num_classes": 2}
TEST_CASE_1 = [senet154, net_args]
TEST_CASE_2 = [se_resnet50, net_args]
TEST_CASE_3 = [se_resnet101, net_args]
TEST_CASE_4 = [se_resnet152, net_args]
TEST_CASE_5 = [se_resnext50_32x4d, net_args]
TEST_CASE_6 = [se_resnext101_32x4d, net_args]

TEST_CASE_PRETRAINED = [se_resnet50, {"spatial_dims": 2, "in_channels": 3, "num_classes": 2, "pretrained": True}]


class TestSENET(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6])
    def test_senet_shape(self, net, net_args):
        input_data = torch.randn(2, 2, 64, 64, 64).to(device)
        expected_shape = (2, 2)
        net = net(**net_args)
        net = net.to(device).eval()
        with torch.no_grad():
            result = net(input_data)
            self.assertEqual(result.shape, expected_shape)

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6])
    def test_script(self, net, net_args):
        net = net(**net_args)
        input_data = torch.randn(2, 2, 64, 64, 64)
        test_script_save(net, input_data)


class TestPretrainedSENET(unittest.TestCase):
    @parameterized.expand(
        [
            TEST_CASE_PRETRAINED,
        ]
    )
    def test_senet_shape(self, model, input_param):
        net = test_pretrained_networks(model, input_param, device)
        input_data = torch.randn(3, 3, 64, 64).to(device)
        expected_shape = (3, 2)
        net = net.to(device).eval()
        with torch.no_grad():
            result = net(input_data)
            self.assertEqual(result.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
