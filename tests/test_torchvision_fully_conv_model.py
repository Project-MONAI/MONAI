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

import unittest
from unittest import skipUnless

import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.nets import TorchVisionFullyConvModel
from monai.utils import optional_import

_, has_tv = optional_import("torchvision")

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASE_0 = [{"model_name": "resnet18", "num_classes": 1, "pretrained": False}, (2, 3, 224, 224), (2, 1, 1, 1)]

TEST_CASE_1 = [{"model_name": "resnet18", "num_classes": 1, "pretrained": False}, (2, 3, 256, 256), (2, 1, 2, 2)]

TEST_CASE_2 = [{"model_name": "resnet101", "num_classes": 5, "pretrained": False}, (2, 3, 256, 256), (2, 5, 2, 2)]

TEST_CASE_3 = [
    {"model_name": "resnet101", "num_classes": 5, "pool_size": 6, "pretrained": False},
    (2, 3, 224, 224),
    (2, 5, 2, 2),
]

TEST_CASE_PRETRAINED_0 = [
    {"model_name": "resnet18", "num_classes": 1, "pretrained": True},
    (2, 3, 224, 224),
    (2, 1, 1, 1),
    -0.010419349186122417,
]

TEST_CASE_PRETRAINED_1 = [
    {"model_name": "resnet18", "num_classes": 1, "pretrained": True},
    (2, 3, 256, 256),
    (2, 1, 2, 2),
    -0.010419349186122417,
]

TEST_CASE_PRETRAINED_2 = [
    {"model_name": "resnet18", "num_classes": 5, "pretrained": True},
    (2, 3, 256, 256),
    (2, 5, 2, 2),
    -0.010419349186122417,
]


class TestTorchVisionFullyConvModel(unittest.TestCase):
    @parameterized.expand([TEST_CASE_0, TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    @skipUnless(has_tv, "Requires TorchVision.")
    def test_without_pretrained(self, input_param, input_shape, expected_shape):
        net = TorchVisionFullyConvModel(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)

    @parameterized.expand([TEST_CASE_PRETRAINED_0, TEST_CASE_PRETRAINED_1, TEST_CASE_PRETRAINED_2])
    @skipUnless(has_tv, "Requires TorchVision.")
    def test_with_pretrained(self, input_param, input_shape, expected_shape, expected_value):
        net = TorchVisionFullyConvModel(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            value = next(net.parameters())[0, 0, 0, 0].item()
            self.assertEqual(value, expected_value)
            self.assertEqual(result.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
