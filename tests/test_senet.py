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
from typing import TYPE_CHECKING
from unittest import skipUnless

import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.nets import SENet154, SEResNet50, SEResNet101, SEResNet152, SEResNext50, SEResNext101
from monai.utils import optional_import
from tests.utils import test_pretrained_networks, test_script_save

if TYPE_CHECKING:
    import pretrainedmodels

    has_cadene_pretrain = True
else:
    pretrainedmodels, has_cadene_pretrain = optional_import("pretrainedmodels")


device = "cuda" if torch.cuda.is_available() else "cpu"

NET_ARGS = {"spatial_dims": 3, "in_channels": 2, "num_classes": 2}
TEST_CASE_1 = [SENet154, NET_ARGS]
TEST_CASE_2 = [SEResNet50, NET_ARGS]
TEST_CASE_3 = [SEResNet101, NET_ARGS]
TEST_CASE_4 = [SEResNet152, NET_ARGS]
TEST_CASE_5 = [SEResNext50, NET_ARGS]
TEST_CASE_6 = [SEResNext101, NET_ARGS]

TEST_CASE_PRETRAINED_1 = [SEResNet50, {"spatial_dims": 2, "in_channels": 3, "num_classes": 2, "pretrained": True}]


class TestSENET(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6])
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

    @parameterized.expand([TEST_CASE_PRETRAINED_1])
    @skipUnless(has_cadene_pretrain, "Requires `pretrainedmodels` package.")
    def test_pretrain_consistency(self, model, input_param):
        input_data = torch.randn(1, 3, 64, 64).to(device)
        net = test_pretrained_networks(model, input_param, device)
        with eval_mode(net):
            result = net.features(input_data)
        cadene_net = pretrainedmodels.se_resnet50().to(device)
        with eval_mode(cadene_net):
            expected_result = cadene_net.features(input_data)
        # The difference between Cadene's senet and our version is that
        # we use nn.Linear as the FC layer, but Cadene's version uses
        # a conv layer with kernel size equals to 1. It may bring a little difference.
        self.assertTrue(torch.allclose(result, expected_result, rtol=1e-5, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
