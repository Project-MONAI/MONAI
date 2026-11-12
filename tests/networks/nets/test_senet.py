# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from parameterized import parameterized

import monai.networks.nets.senet as se_mod
from monai.networks import eval_mode
from monai.networks.nets import SENet, SENet154, SEResNet50, SEResNet101, SEResNet152, SEResNext50, SEResNext101
from tests.test_utils import test_pretrained_networks, test_script_save, testing_data_config, TESTING_DATA_PATH

device = "cuda" if torch.cuda.is_available() else "cpu"

TESTING_DATA_URLS = {
    "senet154": {
        "url": testing_data_config("models", "senet154-c7b49a05", "url"),
        "filename":TESTING_DATA_PATH / "senet154-c7b49a05.pth",
    },
    "se_resnet50": {
        "url": testing_data_config("models", "se_resnet50-ce0d4300", "url"),
        "filename":TESTING_DATA_PATH / "se_resnet50-ce0d4300.pth",
    },
    "se_resnet101": {
        "url": testing_data_config("models", "se_resnet101-7e38fcc6", "url"),
        "filename":TESTING_DATA_PATH / "se_resnet101-7e38fcc6.pth",
    },
    "se_resnet152": {
        "url": testing_data_config("models", "se_resnet152-d17c99b7", "url"),
        "filename": TESTING_DATA_PATH /"se_resnet152-d17c99b7.pth",
    },
    "se_resnext50_32x4d": {
        "url": testing_data_config("models", "se_resnext50_32x4d-a260b3a4", "url"),
        "filename": TESTING_DATA_PATH /"se_resnext50_32x4d-a260b3a4.pth",
    },
    "se_resnext101_32x4d": {
        "url": testing_data_config("models", "se_resnext101_32x4d-3b2fe3d8", "url"),
        "filename": TESTING_DATA_PATH /"se_resnext101_32x4d-3b2fe3d8.pth",
    },
}


NET_ARGS = {"spatial_dims": 3, "in_channels": 2, "num_classes": 2}
TEST_CASE_1 = [SENet154, NET_ARGS]
TEST_CASE_2 = [SEResNet50, NET_ARGS]
TEST_CASE_3 = [SEResNet101, NET_ARGS]
TEST_CASE_4 = [SEResNet152, NET_ARGS]
TEST_CASE_5 = [SEResNext50, NET_ARGS]
TEST_CASE_6 = [SEResNext101, NET_ARGS]
TEST_CASE_7 = [
    SENet,
    {
        "spatial_dims": 3,
        "in_channels": 2,
        "num_classes": 2,
        "block": "se_bottleneck",
        "layers": (3, 8, 36, 3),
        "groups": 64,
        "reduction": 16,
    },
]

TEST_CASE_PRETRAINED_1 = [SEResNet50, {"spatial_dims": 2, "in_channels": 3, "num_classes": 2, "pretrained": True}]


class TestSENET(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6, TEST_CASE_7])
    def test_senet_shape(self, net, net_args):
        input_data = torch.randn(2, 2, 64, 64, 64).to(device)
        expected_shape = (2, 2)
        net = net(**net_args).to(device)
        with eval_mode(net):
            result = net(input_data)
            self.assertEqual(result.shape, expected_shape)

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6])
    def test_script(self, net, net_args):
        net = net(**net_args)
        input_data = torch.randn(2, 2, 64, 64, 64)
        test_script_save(net, input_data)


@patch.dict(se_mod.SE_NET_MODELS,TESTING_DATA_URLS)
class TestPretrainedSENET(unittest.TestCase):
    @parameterized.expand([TEST_CASE_PRETRAINED_1])
    def test_senet_shape(self, model, input_param):
        net = test_pretrained_networks(model, input_param, device)
        input_data = torch.randn(3, 3, 64, 64).to(device)
        expected_shape = (3, 2)
        net = net.to(device)
        with eval_mode(net):
            result = net(input_data)
            self.assertEqual(result.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
