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
from typing import TYPE_CHECKING

import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.nets import ResNet, resnet10, resnet18, resnet34, resnet50, resnet101, resnet152, resnet200
from monai.networks.nets.resnet import ResNetBlock
from monai.utils import optional_import
from tests.utils import test_script_save

if TYPE_CHECKING:
    import torchvision

    has_torchvision = True
else:
    torchvision, has_torchvision = optional_import("torchvision")

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASE_1 = [  # 3D, batch 3, 2 input channel
    {
        "pretrained": False,
        "spatial_dims": 3,
        "n_input_channels": 2,
        "num_classes": 3,
        "conv1_t_size": 7,
        "conv1_t_stride": (2, 2, 2),
    },
    (3, 2, 32, 64, 48),
    (3, 3),
]

TEST_CASE_2 = [  # 2D, batch 2, 1 input channel
    {
        "pretrained": False,
        "spatial_dims": 2,
        "n_input_channels": 1,
        "num_classes": 3,
        "conv1_t_size": [7, 7],
        "conv1_t_stride": [2, 2],
    },
    (2, 1, 32, 64),
    (2, 3),
]

TEST_CASE_2_A = [  # 2D, batch 2, 1 input channel, shortcut type A
    {
        "pretrained": False,
        "spatial_dims": 2,
        "n_input_channels": 1,
        "num_classes": 3,
        "shortcut_type": "A",
        "conv1_t_size": (7, 7),
        "conv1_t_stride": 2,
    },
    (2, 1, 32, 64),
    (2, 3),
]

TEST_CASE_3 = [  # 1D, batch 1, 2 input channels
    {
        "pretrained": False,
        "spatial_dims": 1,
        "n_input_channels": 2,
        "num_classes": 3,
        "conv1_t_size": [3],
        "conv1_t_stride": 1,
    },
    (1, 2, 32),
    (1, 3),
]

TEST_CASE_3_A = [  # 1D, batch 1, 2 input channels
    {"pretrained": False, "spatial_dims": 1, "n_input_channels": 2, "num_classes": 3, "shortcut_type": "A"},
    (1, 2, 32),
    (1, 3),
]

TEST_CASE_4 = [  # 2D, batch 2, 1 input channel
    {"pretrained": False, "spatial_dims": 2, "n_input_channels": 1, "num_classes": 3, "feed_forward": False},
    (2, 1, 32, 64),
    ((2, 512), (2, 2048)),
]

TEST_CASE_5 = [  # 1D, batch 1, 2 input channels
    {
        "block": "basic",
        "layers": [1, 1, 1, 1],
        "block_inplanes": [64, 128, 256, 512],
        "spatial_dims": 1,
        "n_input_channels": 2,
        "num_classes": 3,
        "conv1_t_size": [3],
        "conv1_t_stride": 1,
    },
    (1, 2, 32),
    (1, 3),
]

TEST_CASE_5_A = [  # 1D, batch 1, 2 input channels
    {
        "block": ResNetBlock,
        "layers": [1, 1, 1, 1],
        "block_inplanes": [64, 128, 256, 512],
        "spatial_dims": 1,
        "n_input_channels": 2,
        "num_classes": 3,
        "conv1_t_size": [3],
        "conv1_t_stride": 1,
    },
    (1, 2, 32),
    (1, 3),
]

TEST_CASE_6 = [  # 1D, batch 1, 2 input channels
    {
        "block": "bottleneck",
        "layers": [3, 4, 6, 3],
        "block_inplanes": [64, 128, 256, 512],
        "spatial_dims": 1,
        "n_input_channels": 2,
        "num_classes": 3,
        "conv1_t_size": [3],
        "conv1_t_stride": 1,
    },
    (1, 2, 32),
    (1, 3),
]

TEST_CASES = []
for case in [TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_2_A, TEST_CASE_3_A]:
    for model in [resnet10, resnet18, resnet34, resnet50, resnet101, resnet152, resnet200]:
        TEST_CASES.append([model, *case])
for case in [TEST_CASE_5, TEST_CASE_5_A, TEST_CASE_6]:
    TEST_CASES.append([ResNet, *case])

TEST_SCRIPT_CASES = [
    [model, *TEST_CASE_1] for model in [resnet10, resnet18, resnet34, resnet50, resnet101, resnet152, resnet200]
]


class TestResNet(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_resnet_shape(self, model, input_param, input_shape, expected_shape):
        net = model(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            if input_param.get("feed_forward", True):
                self.assertEqual(result.shape, expected_shape)
            else:
                self.assertTrue(result.shape in expected_shape)

    @parameterized.expand(TEST_SCRIPT_CASES)
    def test_script(self, model, input_param, input_shape, expected_shape):
        net = model(**input_param)
        test_data = torch.randn(input_shape)
        test_script_save(net, test_data)


if __name__ == "__main__":
    unittest.main()
